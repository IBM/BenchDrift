"""
BenchDrift App — HuggingFace dataset loader + JSON upload.

Handles loading problems from HuggingFace datasets and custom JSON/JSONL files.
"""

import json
import re
from typing import List, Tuple

import gradio as gr

from app.ollama import call_llm

try:
    import datasets as hf_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# In-memory cache: (dataset_name, config, split) -> list of row dicts
_hf_cache: dict = {}


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _get_real_configs(dataset_name: str) -> list:
    """Get the actual loadable configs for a dataset (not metadata keys).

    Returns empty list if the dataset doesn't exist or can't be accessed,
    so callers can fall through to fuzzy search.
    """
    try:
        configs = hf_datasets.get_dataset_config_names(dataset_name, trust_remote_code=True)
        return configs if configs else ["default"]
    except Exception:
        return []


def _get_builder_info(dataset_name: str, config_name: str = None):
    """Get splits and columns from load_dataset_builder (always accurate)."""
    try:
        builder = hf_datasets.load_dataset_builder(
            dataset_name, config_name, trust_remote_code=True,
        )
        splits = list(builder.info.splits.keys()) if builder.info.splits else ["train", "test"]
        columns = list(builder.info.features.keys()) if builder.info.features else []
        return splits, columns
    except Exception:
        # Builder with config=None may fail if dataset requires a config.
        # Fall back to get_dataset_infos for the first available config.
        if config_name is None:
            try:
                infos = hf_datasets.get_dataset_infos(dataset_name)
                if infos:
                    first_info = next(iter(infos.values()))
                    splits = list(first_info.splits.keys()) if first_info.splits else ["train", "test"]
                    columns = list(first_info.features.keys()) if first_info.features else []
                    return splits, columns
            except Exception:
                pass
        return ["train", "test"], []


def hf_load_dataset_info(dataset_name: str):
    """Load dataset metadata (configs, splits, columns) without downloading data.

    Returns 8-tuple: (config_update, split_update, problem_col_update,
                       answer_col_update, problem_selector_update,
                       rows_state, status_html, dataset_name_update).

    The 8th value writes the resolved dataset name back into the textbox,
    so that hf_config.change sees the real HF name (not the user's search term).
    """
    _empty = (
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        [],
    )
    if not HF_AVAILABLE:
        return (*_empty, "pip install datasets to enable HuggingFace loading", gr.update())
    if not dataset_name or not dataset_name.strip():
        return (*_empty, "", gr.update())

    dataset_name = dataset_name.strip()
    resolved_name = dataset_name
    configs = _get_real_configs(dataset_name)

    if not configs:
        # Exact name failed — search HF Hub
        try:
            from huggingface_hub import list_datasets
            matches = list(list_datasets(search=dataset_name, limit=15))
            if matches:
                scored = []
                for m in matches:
                    mid = m.id.lower()
                    term = dataset_name.lower()
                    if term in mid:
                        scored.append((0, m.id))
                    elif any(w in mid for w in term.split()):
                        scored.append((1, m.id))
                    else:
                        scored.append((2, m.id))
                scored.sort()
                match_ids = [s[1] for s in scored]
                resolved_name = match_ids[0]
                configs = _get_real_configs(resolved_name)
                if not configs:
                    # Best match also failed — show fuzzy results for user to pick
                    return (
                        gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None),
                        gr.update(choices=match_ids, value=match_ids[0]),
                        [],
                        f"Found {len(match_ids)} datasets — pick one from dropdown, then click Load again: {', '.join(match_ids[:5])}",
                        gr.update(),
                    )
        except Exception:
            pass

    if not configs:
        return (*_empty, f"No dataset found for '{dataset_name}'", gr.update())

    first_config = configs[0]
    # Use load_dataset_builder for accurate splits/columns
    splits, columns = _get_builder_info(resolved_name, first_config if first_config != "default" else None)

    problem_guess, answer_guess = _guess_columns(columns)

    if problem_guess and answer_guess:
        status = f"loaded {resolved_name} — auto-detected: problem=<b>{_esc(problem_guess)}</b>, answer=<b>{_esc(answer_guess)}</b>"
    else:
        missing = []
        if not problem_guess:
            missing.append("problem")
        if not answer_guess:
            missing.append("answer")
        status = (
            f"loaded {resolved_name} — columns: {', '.join(columns[:10])}<br>"
            f"<b>could not auto-detect {' or '.join(missing)} column — please select manually</b>"
        )

    return (
        gr.update(choices=configs, value=first_config),
        gr.update(choices=splits, value=splits[0] if splits else None),
        gr.update(choices=columns, value=problem_guess),
        gr.update(choices=columns, value=answer_guess),
        gr.update(choices=[], value=None),
        [],
        status,
        gr.update(value=resolved_name),  # Write resolved name back to textbox
    )


def hf_on_config_change(dataset_name: str, config_name: str):
    """When user changes config, update splits and columns."""
    if not HF_AVAILABLE or not dataset_name or not config_name:
        return gr.update(), gr.update(), gr.update(), gr.update(), [], ""
    try:
        cfg = config_name if config_name and config_name != "default" else None
        splits, columns = _get_builder_info(dataset_name.strip(), cfg)
        problem_guess, answer_guess = _guess_columns(columns)
        return (
            gr.update(choices=splits, value=splits[0] if splits else None),
            gr.update(choices=columns, value=problem_guess),
            gr.update(choices=columns, value=answer_guess),
            gr.update(choices=[], value=None),
            [],
            f"config '{config_name}': columns: {', '.join(columns[:8])}",
        )
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), gr.update(), [], f"Error: {e}"


def _hf_load_rows(resolved: str, config_name: str, split_name: str, n: int = 50) -> list:
    cache_key = (resolved, config_name or "", split_name)
    if cache_key in _hf_cache:
        return _hf_cache[cache_key][:n]

    cfg = config_name if config_name and config_name != "default" else None
    try:
        ds = hf_datasets.load_dataset(
            resolved, cfg,
            split=split_name, streaming=True, trust_remote_code=True,
        )
    except (ValueError, FileNotFoundError) as e:
        if "BuilderConfig" in str(e) and cfg is not None:
            # Config from metadata doesn't match actual builder configs — retry without it
            ds = hf_datasets.load_dataset(
                resolved, None,
                split=split_name, streaming=True, trust_remote_code=True,
            )
        else:
            raise

    rows = []
    for row in ds:
        rows.append(dict(row))
        if len(rows) >= n:
            break
    _hf_cache[cache_key] = rows
    return rows


def hf_fetch_problems(dataset_name: str, config_name: str, split_name: str,
                       problem_col: str, answer_col: str, search_query: str,
                       n: int = 50):
    """Fetch first n rows and populate problem dropdown."""
    if not HF_AVAILABLE:
        return gr.update(choices=[], value=None), [], "", "pip install datasets"
    if not dataset_name or not split_name or not problem_col:
        return gr.update(choices=[], value=None), [], "", "Select dataset, split, and problem column"

    resolved = dataset_name.strip()
    try:
        rows = _hf_load_rows(resolved, config_name, split_name, n=n)
    except Exception:
        try:
            from huggingface_hub import list_datasets
            matches = list(list_datasets(search=dataset_name.strip(), limit=5))
            for m in matches:
                if dataset_name.strip().lower() in m.id.lower():
                    resolved = m.id
                    break
            else:
                if matches:
                    resolved = matches[0].id
            rows = _hf_load_rows(resolved, config_name, split_name, n=n)
        except Exception as e:
            return gr.update(choices=[], value=None), [], "", f"Error loading: {e}"
    if not rows:
        return gr.update(choices=[], value=None), [], "", "No rows found"

    choices = []
    for i, row in enumerate(rows):
        text = str(row.get(problem_col, ""))[:80].replace("\n", " ")
        if search_query and search_query.strip():
            if search_query.strip().lower() not in text.lower():
                continue
        choices.append(f"[{i}] {text}")

    ans_instruction = hf_infer_answer_instruction(rows, answer_col)
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        rows,
        ans_instruction,
        f"{len(choices)} problems loaded",
    )


def hf_infer_answer_instruction(rows: list, ans_col: str, gen_model: str = "", backend: str = "ollama") -> str:
    """Infer the expected answer format from sample ground truth answers."""
    if not rows or not ans_col:
        return ""

    samples = []
    for row in rows[:5]:
        if ans_col in row:
            samples.append(str(row[ans_col]))
    if not samples:
        return ""

    has_opts = any(row.get('choices') or row.get('options') for row in rows[:5])
    all_single_letters = all(len(s.strip()) == 1 and s.strip().isalpha() for s in samples)
    all_single_digits = all(s.strip().isdigit() and len(s.strip()) <= 2 for s in samples)
    has_hash = any('####' in s for s in samples)
    has_boxed = any('\\boxed' in s for s in samples)
    all_numeric = all(re.match(r'^-?\d+\.?\d*$', s.strip()) for s in samples)

    if has_hash:
        return "Answer with ONLY the final numerical answer. No explanation, no work."
    if has_boxed:
        return "Answer with ONLY the final answer. No explanation, no work."
    if all_single_letters and has_opts:
        return "This is a multiple-choice question. Answer with ONLY the letter of the correct option (e.g., A, B, C, D). Nothing else."
    if all_single_digits and has_opts:
        return "This is a multiple-choice question. Answer with ONLY the letter of the correct option (e.g., A, B, C, D). Nothing else."
    if all_numeric:
        return "Answer with ONLY the final numerical answer. No explanation, no work."

    if gen_model and not gen_model.startswith("("):
        sample_str = "\n".join(f"  GT answer {i+1}: {s[:100]}" for i, s in enumerate(samples[:3]))
        try:
            resp = call_llm(
                gen_model,
                "You determine answer formats for datasets. Reply with ONLY a short instruction sentence.",
                f"These are sample ground truth answers from a dataset:\n{sample_str}\n\n"
                f"{'The questions have multiple-choice options.' if has_opts else ''}\n"
                f"Write ONE clear instruction sentence telling a model exactly how to format its answer "
                f"to match this format. Be specific about what to include and what NOT to include.",
                backend=backend, max_tokens=100, temperature=0.0, think=False,
            )
            if resp and len(resp) < 200:
                return resp.strip().rstrip('.') + "."
        except Exception:
            pass

    if has_opts:
        return "This is a multiple-choice question. Answer with ONLY the letter of the correct option. Nothing else."
    return "Answer with ONLY the final answer. No explanation, no reasoning, no work shown."


def hf_parse_answer(row: dict, ans_col: str) -> str:
    """Parse answer from a HF dataset row, handling common formats."""
    if not ans_col or ans_col not in row:
        return ""
    raw = str(row[ans_col])

    m = re.search(r'####\s*(.+)', raw)
    if m:
        return m.group(1).strip()
    m = re.search(r'\\boxed\{(.+?)\}', raw)
    if m:
        return m.group(1).strip()

    opts = row.get('choices', row.get('options', []))
    if not isinstance(opts, list):
        opts = []

    if raw.strip().isdigit() and opts:
        idx = int(raw.strip())
        return chr(65 + idx)

    letter = raw.strip().upper()
    if len(letter) == 1 and letter.isalpha() and opts:
        return letter

    return raw.strip()


def hf_select_problem(choice: str, rows: list, problem_col: str, answer_col: str,
                       answer_instruction: str = ""):
    """When user picks a problem from dropdown, fill inputs."""
    if not choice or not rows:
        return "", ""
    m = re.match(r'\[(\d+)\]', choice)
    if not m:
        return "", ""
    idx = int(m.group(1))
    if idx >= len(rows):
        return "", ""

    row = rows[idx]
    problem = str(row.get(problem_col, ""))

    opts = row.get('choices', row.get('options', []))
    if isinstance(opts, list) and opts:
        problem += "\n"
        for i, opt in enumerate(opts):
            letter = chr(65 + i)
            problem += f"\n{letter}) {opt}"

    if answer_instruction:
        problem += f"\n\n[Instruction: {answer_instruction}]"

    answer = hf_parse_answer(row, answer_col) if answer_col else ""
    return problem, answer


def json_upload_handler(file_obj):
    """Parse uploaded JSON/JSONL file and populate problem selector."""
    if file_obj is None:
        return gr.update(choices=[], value=None), [], "", "", ""

    try:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        return gr.update(choices=[], value=None), [], "", "", f"Error reading file: {e}"

    rows = []
    try:
        if text.startswith("["):
            rows = json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except json.JSONDecodeError as e:
        return gr.update(choices=[], value=None), [], "", "", f"JSON parse error: {e}"

    if not rows or not isinstance(rows, list) or not isinstance(rows[0], dict):
        return gr.update(choices=[], value=None), [], "", "", "Expected array of objects or JSONL"

    columns = list(rows[0].keys())
    problem_col, answer_col = _guess_columns(columns)

    if not problem_col and columns:
        problem_col = columns[0]
    if not answer_col and len(columns) > 1:
        answer_col = columns[1]

    choices = []
    for i, row in enumerate(rows):
        text_preview = str(row.get(problem_col, ""))[:80].replace("\n", " ")
        choices.append(f"[{i}] {text_preview}")

    status = (
        f"loaded {len(rows)} problems from upload "
        f"(problem: <b>{_esc(problem_col)}</b>, answer: <b>{_esc(answer_col)}</b>)"
    )
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        rows, problem_col, answer_col, status,
    )


def _guess_columns(columns: list) -> Tuple[str, str]:
    """Auto-detect problem/answer columns from column names.

    Returns (problem_col, answer_col) — None for either if not found.
    """
    problem_guess = None
    answer_guess = None

    # Exact matches (high confidence)
    problem_exact = {"question", "problem", "input", "prompt", "text", "query",
                     "instruction", "statement", "stem"}
    answer_exact = {"answer", "expected", "target", "output", "label", "solution",
                    "response", "correct_answer", "ground_truth", "gold"}

    for c in columns:
        cl = c.lower().strip()
        if cl in problem_exact and not problem_guess:
            problem_guess = c
        if cl in answer_exact and not answer_guess:
            answer_guess = c

    # Substring matches (lower confidence) if exact didn't work
    if not problem_guess:
        for c in columns:
            cl = c.lower()
            if any(k in cl for k in ("question", "problem", "prompt", "input", "instruction")) and not problem_guess:
                problem_guess = c
    if not answer_guess:
        for c in columns:
            cl = c.lower()
            if any(k in cl for k in ("answer", "target", "solution", "label", "output")) and not answer_guess:
                answer_guess = c

    return problem_guess, answer_guess
