"""
BenchDrift App — Unified run engine.

Single streaming generator with RunMode enum that handles all execution paths:
FULL, RETEST, REEVAL, INCREMENTAL.

Uses the pipeline's own answer matching and judging — no duplicate logic:
- StringBasedMatcher from benchdrift.eval.llm_answer_matcher
- LLMJudgeAnswerChecker from benchdrift.eval.llm_judge_answer_checker
- Prompts from benchdrift.pipeline.prompts
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from benchdrift.eval.answer_normalization import normalize_for_judge, parse_judge_response
from benchdrift.eval.llm_answer_matcher import StringBasedMatcher
from benchdrift.pipeline.comprehensive_variation_engine_v2 import (
    clean_model_response,
    is_valid_question,
)
from benchdrift.pipeline.feature_discovery import discover_novel_features
from benchdrift.pipeline.feature_relevance import (
    TRANSFORMATION_TO_AXIS,
    find_unmapped_features,
    get_problem_features,
    rank_transformations_two_level,
)
from benchdrift.pipeline.freeform_variation_engine import FreeformVariationEngine
from benchdrift.pipeline.prompts import VALIDATION_SYSTEM, VALIDATION_USER
from benchdrift.pipeline.run_strategies import (
    RunMode, build_meta, parse_prev, inject_meta,
    generation_changed, testing_changed, eval_changed,
)
from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine

from app.ollama import call_llm, clean_ollama_response, clean_think_tags
from app.renderers import (
    _filter_types_by_axes,
    DEFAULT_AXES_LIST,
    render_baseline,
    render_cards,
    render_drift_chart,
    render_summary,
)

ALL_TRANSFORMATION_TYPES = UnifiedVariationEngine.get_all_transformation_types()
GENERIC_TYPES = {k: v for k, v in ALL_TRANSFORMATION_TYPES.items() if not k.endswith("_persona")}

# Detection engine (no model client needed — entity detection only)
_DETECTION_ENGINE = UnifiedVariationEngine(model_client=None)

# Cached freeform engine instances by model name
_freeform_cache = {}


def _get_freeform_engine(gen_model: str, cfg: dict) -> FreeformVariationEngine:
    """Get or create a cached FreeformVariationEngine for the given model."""
    cache_key = (gen_model, cfg.get("backend", "ollama"), cfg.get("ollama_base_url", ""))
    if cache_key not in _freeform_cache:
        from app.ollama import _get_client
        client = _get_client(cfg.get("backend", "ollama"), gen_model,
                             cfg.get("ollama_base_url", ""),
                             int(cfg.get("ollama_timeout", 0)))
        _freeform_cache[cache_key] = FreeformVariationEngine(model_client=client)
    return _freeform_cache[cache_key]


# Reuse pipeline's string matcher singleton
_STRING_MATCHER = StringBasedMatcher()


# ---------------------------------------------------------------------------
# Answer checking — delegates to pipeline's own logic
# ---------------------------------------------------------------------------

def check_answer(predicted: str, truth: str, cfg: dict) -> bool:
    """Check answer using pipeline's StringBasedMatcher or LLM judge.

    For LLM judge, uses the pipeline's LLMJudgeAnswerChecker via OllamaClient.
    """
    if cfg.get("eval_method") == "llm judge":
        return _llm_judge_check(predicted, truth, cfg)
    result = _STRING_MATCHER.string_match_answers(truth, predicted)
    return result["is_correct"]


def _llm_judge_check(predicted: str, truth: str, cfg: dict) -> bool:
    """LLM judge using pipeline's judge checker via direct Ollama call.

    Uses category-aware few-shot examples from judge_few_shots module.
    Category is detected from the expected answer + optional dataset name (zero LLM cost).
    """
    # Fast path: string match first
    result = _STRING_MATCHER.string_match_answers(truth, predicted)
    if result["is_correct"]:
        return True

    judge_model = cfg.get("judge_model") or cfg.get("_gen_model", "")
    if not judge_model or judge_model.startswith("("):
        return result["is_correct"]

    # Normalize verbose ground truth (MATH dataset, etc.)
    norm_truth = normalize_for_judge(truth)

    # Get category-specific few-shot examples (zero cost — regex + dataset name)
    # If cfg["llm_trait_fallback"] is True, falls back to LLM when regex misses.
    from benchdrift.eval.judge_few_shots import get_few_shot_examples
    dataset_name = cfg.get("dataset_name", "")

    # Build a backend-agnostic callable for LLM trait fallback
    trait_call_fn = None
    if cfg.get("llm_trait_fallback"):
        def trait_call_fn(prompt: str) -> str:
            return call_llm(
                judge_model, "", prompt,
                backend=cfg.get("backend", "ollama"),
                max_tokens=30, temperature=0.0, think=False,
                base_url=cfg.get("ollama_base_url", ""),
                timeout=cfg.get("ollama_timeout", 0),
            )

    few_shots = get_few_shot_examples(norm_truth, dataset_name, cfg=cfg,
                                      call_fn=trait_call_fn)

    # Build judge prompt with category-aware few-shots
    from benchdrift.eval.llm_judge_answer_checker import LLMJudgeAnswerChecker
    prompt = LLMJudgeAnswerChecker.create_judge_prompt(
        None, predicted, norm_truth, few_shot_examples=few_shots
    )

    try:
        raw = call_llm(
            judge_model, "", prompt,
            backend=cfg.get("backend", "ollama"),
            max_tokens=10, temperature=0.0, think=False,
            base_url=cfg.get("ollama_base_url", ""),
            timeout=cfg.get("ollama_timeout", 0),
        )
        judgment = parse_judge_response(raw)
        if judgment is not None:
            return judgment
    except Exception:
        pass
    return result["is_correct"]


# ---------------------------------------------------------------------------
# Variation generation + validation — uses pipeline prompts
# ---------------------------------------------------------------------------

def validate_variation(model: str, original: str, variation: str, cfg: dict) -> bool:
    """Validate using pipeline's VALIDATION_SYSTEM/VALIDATION_USER prompts."""
    user_prompt = VALIDATION_USER.format(original=original, variation=variation)
    try:
        raw = call_llm(
            model, VALIDATION_SYSTEM, user_prompt,
            backend=cfg.get("backend", "ollama"),
            max_tokens=10, temperature=0.0, think=False,
            base_url=cfg.get("ollama_base_url", ""),
            timeout=int(cfg.get("ollama_timeout", 0)),
        )
        resp = raw.upper().strip()
        if not resp:
            return False
        if "VALID" in resp and "INVALID" not in resp:
            return True
        return False
    except Exception:
        return False


def generate_one_variation(model: str, problem: str, trans_name: str,
                           config: dict, cfg: dict) -> str:
    """Generate a single variation via Ollama using pipeline's prompt structure."""
    max_retries = int(cfg.get("gen_max_retries", 2))
    do_validate = bool(cfg.get("validate_variations", False))

    # Use pipeline's GENERIC_VARIATION prompt structure from prompts.py
    system_prompt = (
        f"You are an expert at creating intent-preserving question variations.\n\n"
        f"TASK: Create a {trans_name} variation of the given problem.\n\n"
        f"TRANSFORMATION GOAL: {config['prompt']}\n\n"
        f"UNIVERSAL RULES (apply to ALL transformations):\n"
        f"1. PRESERVE the exact answer — numerical value, letter choice, or text\n"
        f"2. MAINTAIN all mathematical/logical relationships\n"
        f"3. PRESERVE structural format of the question:\n"
        f"   - If the question has multiple-choice options (A/B/C/D), keep them\n"
        f"   - If it has fill-in-the-blank (___), keep the blanks\n"
        f"   - If it has bullet points or numbered lists, keep the format\n"
        f"   - If it has tables or structured data, keep the layout\n"
        f"4. Numbers: format can change (5 -> five), value CANNOT (5 -> 6)\n"
        f"5. Units: convert correctly or not at all (60 miles -> 96.5 km, NOT 60 km)\n"
        f"6. Time points stay time points, durations stay durations\n"
        f"7. Use PLAIN TEXT only — no markdown formatting\n"
        f"8. Make the variation as linguistically different as possible\n"
        f"9. Return ONLY the transformed question inside <question> tags\n"
        f"10. Do NOT explain your reasoning\n\n"
        f"<question>Your transformed question here</question>"
    )
    user_prompt = f"Original: {problem}\n\nReturn only the <question>...</question>."

    last_cleaned = ""
    for attempt in range(max_retries + 1):
        raw = call_llm(model, system_prompt, user_prompt,
                       backend=cfg.get("backend", "ollama"),
                       max_tokens=int(cfg.get("gen_max_tokens", 1024)),
                       temperature=float(cfg.get("gen_temperature", 0.5)),
                       think=False,
                       base_url=cfg.get("ollama_base_url", ""),
                       timeout=int(cfg.get("ollama_timeout", 0)))
        cleaned = clean_ollama_response(raw)
        if cleaned and cleaned.strip().upper() == "SKIP":
            return "SKIP"
        if not cleaned or not is_valid_question(cleaned):
            last_cleaned = cleaned or ""
            if attempt < max_retries:
                user_prompt = (
                    f"Original: {problem}\n\n"
                    f"You MUST return ONLY the question text inside <question> tags. "
                    f"No explanation, no analysis, no reasoning. Just the question."
                )
            continue

        if do_validate:
            is_valid = validate_variation(model, problem, cleaned, cfg=cfg)
            if not is_valid:
                last_cleaned = cleaned
                if attempt < max_retries:
                    user_prompt = (
                        f"Original: {problem}\n\n"
                        f"Your previous variation was INVALID — it changed the answer. "
                        f"Create a new {trans_name} variation that preserves the EXACT same answer. "
                        f"Return ONLY the question inside <question> tags."
                    )
                continue

        return cleaned

    if do_validate and last_cleaned and is_valid_question(last_cleaned):
        return "INVALID"
    return last_cleaned or ""


# ---------------------------------------------------------------------------
# Helpers: instruction tags, MCQ stripping, referential detection
# ---------------------------------------------------------------------------

def strip_instruction_tag(problem: str) -> Tuple[str, str]:
    m = re.search(r'\n*\[Instruction:\s*(.+?)\]\s*$', problem, re.DOTALL)
    if m:
        instruction = m.group(1).strip()
        clean = problem[:m.start()].rstrip()
        return clean, instruction
    return problem, ""


def _strip_mcq_options(problem: str) -> str:
    m = re.search(r'\n\s*A[).]\s', problem)
    if m:
        after = problem[m.start():]
        if re.search(r'\bB[).]\s', after):
            return problem[:m.start()].rstrip()
    return problem


def detect_referential_candidates(problem: str) -> list:
    try:
        stem = _strip_mcq_options(problem)
        candidates = _DETECTION_ENGINE._detect_all_candidates_with_composites(stem)
        merged = _DETECTION_ENGINE._detect_and_merge_dependencies(stem, candidates)
        return merged
    except Exception:
        return []


def _get_varied_cluster(problem: str, variant: str) -> str:
    cands = detect_referential_candidates(problem)
    if not cands or not variant:
        return ""
    varied = []
    for c in cands:
        t = c.get('text', '').strip()
        if t and t.lower() not in variant.lower():
            varied.append(t)
    if varied:
        return "+".join(varied[:4])
    for c in cands:
        t = c.get('text', '').strip()
        if t:
            return t
    return ""


def _auto_discover_features(problem_text: str, model_name: str,
                             existing_features: dict,
                             cfg: dict = None) -> tuple:
    """Ask LLM to identify novel features not in the predefined set.

    Thin wrapper around benchdrift.pipeline.feature_discovery.discover_novel_features()
    that constructs the call_fn from the app's Ollama backend.
    """
    if not problem_text or not problem_text.strip():
        return {}, ""
    if not model_name or model_name.startswith("("):
        return {}, "Select a model first"
    if cfg is None:
        cfg = {}

    def _call_fn(system_prompt: str, user_prompt: str) -> str:
        return call_llm(model_name, system_prompt, user_prompt,
                        backend=cfg.get("backend", "ollama"),
                        max_tokens=256, temperature=0.3, think=False,
                        base_url=cfg.get("ollama_base_url", ""),
                        timeout=int(cfg.get("ollama_timeout", 0)))

    return discover_novel_features(problem_text, existing_features, call_fn=_call_fn)


# ---------------------------------------------------------------------------
# Metadata tracking for smart re-run
# ---------------------------------------------------------------------------
# build_meta, parse_prev, inject_meta, generation_changed, testing_changed,
# eval_changed are imported from benchdrift.pipeline.run_strategies


# ---------------------------------------------------------------------------
# Core helpers shared across all run modes
# ---------------------------------------------------------------------------

def _run_baseline(target_model: str, clean_problem: str, gt: str,
                  answer_instruction: str, cfg: dict) -> Tuple[str, bool]:
    """Run baseline: solve problem with target model, check answer."""
    enable_reasoning = cfg.get("enable_reasoning", False)
    think_budget = int(cfg.get("think_max_tokens", 0))
    solve_system = answer_instruction if answer_instruction else (
        "Solve the problem. Return ONLY the final numerical answer. "
        "No explanation, no work. Just the number."
    )
    max_tok = int(cfg.get("solver_max_tokens", 256))
    if enable_reasoning and think_budget:
        max_tok = think_budget  # total budget includes thinking

    target_backend = cfg.get("target_backend", cfg.get("backend", "ollama"))
    raw = call_llm(
        target_model, solve_system, clean_problem,
        backend=target_backend,
        max_tokens=max_tok,
        temperature=float(cfg.get("solver_temperature", 0.0)),
        think=enable_reasoning,
        base_url=cfg.get("ollama_base_url", ""),
        timeout=int(cfg.get("ollama_timeout", 0)),
        return_reasoning=enable_reasoning,
    )
    if enable_reasoning and isinstance(raw, dict):
        answer = clean_think_tags(raw["content"])
        correct = check_answer(answer, gt, cfg)
        return answer, correct
    answer = clean_think_tags(raw)
    correct = check_answer(answer, gt, cfg)
    return answer, correct


def _test_one_variation(target_model: str, variant: str, gt: str,
                        answer_instruction: str, cfg: dict) -> dict:
    """Test a single variation against the target model."""
    enable_reasoning = cfg.get("enable_reasoning", False)
    think_budget = int(cfg.get("think_max_tokens", 0))
    solve_system = answer_instruction if answer_instruction else (
        "Solve the problem. Return ONLY the final numerical answer. "
        "No explanation, no work. Just the number."
    )
    max_tok = int(cfg.get("solver_max_tokens", 256))
    if enable_reasoning and think_budget:
        max_tok = think_budget

    target_backend = cfg.get("target_backend", cfg.get("backend", "ollama"))
    raw = call_llm(
        target_model, solve_system, variant,
        backend=target_backend,
        max_tokens=max_tok,
        temperature=float(cfg.get("solver_temperature", 0.0)),
        think=enable_reasoning,
        base_url=cfg.get("ollama_base_url", ""),
        timeout=int(cfg.get("ollama_timeout", 0)),
        return_reasoning=enable_reasoning,
    )
    if enable_reasoning and isinstance(raw, dict):
        answer = clean_think_tags(raw["content"])
        correct = check_answer(answer, gt, cfg)
        return {"answer": answer, "correct": correct, "reasoning": raw["thinking"]}
    answer = clean_think_tags(raw)
    correct = check_answer(answer, gt, cfg)
    return {"answer": answer, "correct": correct}


def _select_transformations(clean_problem: str, enabled_axes: set,
                             regex_features: dict, llm_features: dict,
                             axis_ranking: list, features_sync_json: str,
                             top_k: int) -> list:
    """Select transformations using the pipeline's ranking."""
    trans_types = _filter_types_by_axes(enabled_axes)
    merged_features = dict(regex_features) if regex_features else get_problem_features(clean_problem)
    if llm_features:
        merged_features.update(llm_features)
    try:
        overrides = json.loads(features_sync_json) if features_sync_json else {}
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if isinstance(v, bool):
                    merged_features[k] = v
    except (json.JSONDecodeError, TypeError):
        pass
    pre_axes = axis_ranking if axis_ranking else None
    result_3 = rank_transformations_two_level(
        clean_problem, merged_features, trans_types,
        top_k=top_k,
        pre_ranked_axes=pre_axes,
        enabled_axes=enabled_axes,
    )
    # Sort by actual ranked axis order (from relevance ranking), then by score within each axis.
    # The ranking already returns results in axis-priority order from rank_transformations_two_level,
    # but we re-sort here using the pre_axes order (if available) so the display matches.
    if pre_axes:
        axis_order = {ax: i for i, ax in enumerate(pre_axes)}
    else:
        # Extract order from result itself (first occurrence of each axis)
        axis_order = {}
        for _, _, ax in result_3:
            if ax not in axis_order:
                axis_order[ax] = len(axis_order)
    sorted_result = sorted(result_3, key=lambda x: (axis_order.get(x[2], 99), -x[1]))
    return [(name, score, axis) for name, score, axis in sorted_result]


# ---------------------------------------------------------------------------
# Unified streaming generator
# ---------------------------------------------------------------------------

def run(mode: RunMode, problem: str, answer: str,
        gen_model: str, target_model: str,
        top_k: int = 8, enabled_axes: set = None,
        regex_features: dict = None, llm_features: dict = None,
        axis_ranking: list = None, features_sync_json: str = "",
        prev_results: list = None, cfg: dict = None,
        variation_mode: str = "axes-based"):
    """
    Unified streaming generator for all run modes.

    Args:
        variation_mode: "axes-based" (default) or "free-form" (LLM generates all variations)

    Yields (cards_html, summary_html, drift_chart_html, results_json) at each step.
    """
    if cfg is None:
        cfg = {}
    cfg["_gen_model"] = gen_model

    # Validate inputs
    if not problem or not problem.strip():
        yield ("<p style='color:#888'>Enter a problem above.</p>", "", "", "[]")
        return
    if not gen_model or gen_model.startswith("("):
        yield ("<p style='color:#888'>Select a generator model.</p>", "", "", "[]")
        return
    if not target_model or target_model.startswith("("):
        yield ("<p style='color:#888'>Select a target model.</p>", "", "", "[]")
        return
    if not answer or not answer.strip():
        yield ("<p style='color:#888'>Provide the ground-truth answer.</p>", "", "", "[]")
        return

    problem = problem.strip()
    gt = answer.strip()
    clean_problem, answer_instruction = strip_instruction_tag(problem)

    # --- Run baseline ---
    yield (
        '<div class="baseline-card"><span class="st st-wait">running baseline...</span></div>',
        "", "", "[]",
    )

    try:
        baseline_answer, baseline_correct = _run_baseline(
            target_model, clean_problem, gt, answer_instruction, cfg)
    except Exception as e:
        from app.renderers import _esc
        yield (
            f'<div class="baseline-card"><span class="st st-err">baseline failed: {_esc(str(e))}</span></div>',
            "", "", "[]",
        )
        return

    baseline_html = render_baseline(baseline_answer, baseline_correct, gen_model, target_model)

    # --- REEVAL mode: just re-check existing answers, no LLM calls for variants ---
    if mode == RunMode.REEVAL and prev_results:
        results = []
        for r in prev_results:
            if r.get("status") == "done" and r.get("answer"):
                new_r = dict(r)
                new_r["correct"] = check_answer(r["answer"], gt, cfg)
                new_r["drifted"] = new_r["correct"] != baseline_correct
                new_r["positive_drift"] = not baseline_correct and new_r["correct"]
                new_r["negative_drift"] = baseline_correct and not new_r["correct"]
                results.append(new_r)
            else:
                results.append(dict(r))
        yield (
            baseline_html + render_cards(results),
            render_summary(results, baseline_correct, baseline_answer),
            render_drift_chart(results, baseline_correct),
            json.dumps([r for r in results if r.get("status") in ("done", "error")], indent=2),
        )
        return

    # --- RETEST mode: keep variants, re-run solver on each ---
    if mode == RunMode.RETEST and prev_results:
        results = []
        for r in prev_results:
            if r.get("variant") and r.get("status") in ("done",) and r.get("error") is None:
                results.append({
                    "name": r["name"], "score": r.get("score", 0),
                    "axis": r.get("axis", ""), "ref_keywords": r.get("ref_keywords", ""),
                    "variant": r["variant"], "status": "re-testing",
                })
            elif r.get("status") == "error":
                results.append(dict(r))
        yield from _stream_test_loop(results, baseline_html, baseline_correct,
                                      baseline_answer, target_model, gt,
                                      answer_instruction, cfg)
        return

    # --- FREE-FORM mode: stream one variation at a time ---
    if variation_mode == "free-form" and mode in (RunMode.FULL, RunMode.INCREMENTAL):
        yield (
            baseline_html + '<div class="result-card"><span class="st st-wait">detecting domain &amp; preparing examples...</span></div>',
            "", "", "[]",
        )

        try:
            engine = _get_freeform_engine(gen_model, cfg)
            context = engine.prepare_context(clean_problem)
        except Exception as e:
            from app.renderers import _esc
            yield (
                baseline_html + f'<div class="result-card"><span class="st st-err">free-form setup failed: {_esc(str(e))}</span></div>',
                "", "", "[]",
            )
            return

        do_validate = bool(cfg.get("validate_variations", False))
        results = []
        previous_variants = []
        executor = ThreadPoolExecutor(max_workers=1)
        pending_future = None

        def finalize():
            nonlocal pending_future
            if pending_future is None:
                return
            future, idx = pending_future
            try:
                drift_result = future.result(timeout=130)
                results[idx]["answer"] = drift_result["answer"]
                results[idx]["correct"] = drift_result["correct"]
                results[idx]["drifted"] = drift_result["correct"] != baseline_correct
                results[idx]["positive_drift"] = not baseline_correct and drift_result["correct"]
                results[idx]["negative_drift"] = baseline_correct and not drift_result["correct"]
                if drift_result.get("reasoning"):
                    results[idx]["reasoning"] = drift_result["reasoning"]
                results[idx]["status"] = "done"
            except Exception as e:
                results[idx]["status"] = "error"
                results[idx]["error"] = str(e)
            pending_future = None

        def make_yield():
            return (
                baseline_html + render_cards(results),
                render_summary(results, baseline_correct, baseline_answer),
                render_drift_chart(results, baseline_correct),
                json.dumps([r for r in results if r.get("status") in ("done", "error")], indent=2),
            )

        domain_label = context.get("domain_label", "unknown")
        for i in range(top_k):
            idx = len(results)
            results.append({
                "name": f"freeform_{i+1}", "score": 0.0,
                "axis": "free-form", "ref_keywords": "",
                "status": "generating",
            })
            yield make_yield()

            try:
                v = engine.generate_single_variation(
                    clean_problem, context,
                    previous_variations=previous_variants,
                    validate=do_validate,
                )
            except Exception as e:
                results[idx]["status"] = "error"
                results[idx]["error"] = str(e)
                yield make_yield()
                continue

            if not v or not v.get("modified_problem"):
                results[idx]["status"] = "error"
                results[idx]["error"] = "generation failed"
                yield make_yield()
                continue

            variant_text = v["modified_problem"]
            results[idx]["name"] = v.get("transformation_type", f"freeform_{domain_label}")
            results[idx]["variant"] = variant_text
            results[idx]["status"] = "testing"
            previous_variants.append(variant_text)
            yield make_yield()

            # Finalize previous test before submitting next
            finalize()
            yield make_yield()

            future = executor.submit(
                _test_one_variation, target_model, variant_text, gt, answer_instruction, cfg)
            pending_future = (future, idx)

        finalize()
        executor.shutdown(wait=False)
        yield make_yield()
        return

    # --- FULL or INCREMENTAL: need to select transformations ---
    if enabled_axes is None:
        enabled_axes = set(DEFAULT_AXES_LIST)
    trans_types = _filter_types_by_axes(enabled_axes)
    selected = _select_transformations(
        clean_problem, enabled_axes,
        regex_features or {}, llm_features or {},
        axis_ranking or [], features_sync_json, top_k,
    )

    # For INCREMENTAL, filter out already-completed transformations
    if mode == RunMode.INCREMENTAL and prev_results:
        prev_names = {r["name"] for r in prev_results if r.get("status") in ("done", "error")}
        new_selected = [(n, s, a) for n, s, a in selected if n not in prev_names]

        if not new_selected:
            # Fix stale baseline flags on existing results
            for r in prev_results:
                if r.get("status") == "done":
                    r["drifted"] = r.get("correct", False) != baseline_correct
                    r["positive_drift"] = not baseline_correct and r.get("correct", False)
                    r["negative_drift"] = baseline_correct and not r.get("correct", False)
            yield (
                baseline_html + render_cards(prev_results),
                render_summary(prev_results, baseline_correct, baseline_answer),
                render_drift_chart(prev_results, baseline_correct),
                json.dumps(prev_results, indent=2),
            )
            return

        # Fix stale baseline on existing results
        for r in prev_results:
            if r.get("status") == "done":
                r["drifted"] = r.get("correct", False) != baseline_correct
                r["positive_drift"] = not baseline_correct and r.get("correct", False)
                r["negative_drift"] = baseline_correct and not r.get("correct", False)

        results = list(prev_results)
        selected = new_selected
    else:
        results = []

    # --- Generate + test loop ---
    yield from _stream_generate_and_test(
        results, selected, trans_types, clean_problem, gt,
        gen_model, target_model, answer_instruction,
        baseline_html, baseline_correct, baseline_answer,
        llm_features, cfg,
    )


def _stream_generate_and_test(results, selected, trans_types, clean_problem, gt,
                                gen_model, target_model, answer_instruction,
                                baseline_html, baseline_correct, baseline_answer,
                                llm_features, cfg):
    """Generate variations and test them, streaming results."""
    executor = ThreadPoolExecutor(max_workers=1)
    pending_future = None

    def finalize():
        nonlocal pending_future
        if pending_future is None:
            return
        future, idx = pending_future
        try:
            drift_result = future.result(timeout=130)
            results[idx]["answer"] = drift_result["answer"]
            results[idx]["correct"] = drift_result["correct"]
            results[idx]["drifted"] = drift_result["correct"] != baseline_correct
            results[idx]["positive_drift"] = not baseline_correct and drift_result["correct"]
            results[idx]["negative_drift"] = baseline_correct and not drift_result["correct"]
            if drift_result.get("reasoning"):
                results[idx]["reasoning"] = drift_result["reasoning"]
            results[idx]["status"] = "done"
        except Exception as e:
            results[idx]["status"] = "error"
            results[idx]["error"] = str(e)
        pending_future = None

    def make_yield():
        return (
            baseline_html + render_cards(results),
            render_summary(results, baseline_correct, baseline_answer),
            render_drift_chart(results, baseline_correct),
            json.dumps([r for r in results if r.get("status") in ("done", "error")], indent=2),
        )

    for trans_name, score, axis in selected:
        idx = len(results)
        results.append({"name": trans_name, "score": score, "axis": axis,
                        "ref_keywords": "", "status": "generating"})
        yield make_yield()

        config = trans_types.get(trans_name)
        if not config:
            results[idx]["status"] = "error"
            results[idx]["error"] = "transformation not found"
            yield make_yield()
            continue

        try:
            variant = generate_one_variation(gen_model, clean_problem, trans_name, config, cfg=cfg)
            if not variant:
                results[idx]["status"] = "error"
                results[idx]["error"] = "empty after cleaning + validation"
                yield make_yield()
                continue
            if variant == "SKIP":
                results[idx]["status"] = "error"
                results[idx]["error"] = "skipped — not applicable to this problem"
                yield make_yield()
                continue
            if variant == "INVALID":
                results[idx]["status"] = "error"
                results[idx]["error"] = "validation failed — variant changes the answer"
                yield make_yield()
                continue
        except Exception as e:
            results[idx]["status"] = "error"
            results[idx]["error"] = str(e)
            yield make_yield()
            continue

        results[idx]["variant"] = variant
        if axis == "referential":
            results[idx]["ref_keywords"] = _get_varied_cluster(clean_problem, variant)
        results[idx]["status"] = "testing"
        yield make_yield()

        finalize()
        yield make_yield()

        future = executor.submit(
            _test_one_variation, target_model, variant, gt, answer_instruction, cfg)
        pending_future = (future, idx)

    # Handle unmapped LLM features -> custom variation
    if llm_features:
        selected_names = [name for name, _, _ in selected]
        unmapped = find_unmapped_features(llm_features, selected_names)
        if unmapped:
            feat_name = unmapped[0]
            custom_name = f"custom_{feat_name}"
            idx = len(results)
            results.append({"name": custom_name, "score": 0.0, "axis": "", "status": "generating"})
            yield make_yield()

            custom_system = (
                f"You are an expert at creating question variations.\n\n"
                f"TASK: Create a variation that specifically stress-tests the "
                f"'{feat_name.replace('has_', '').replace('_', ' ')}' aspect of this problem.\n\n"
                f"RULES:\n1. PRESERVE the exact numerical answer\n"
                f"2. MAINTAIN all mathematical relationships\n"
                f"3. Return ONLY the transformed question inside <question> tags\n"
                f"4. Do NOT explain your reasoning\n\n"
                f"<question>Your transformed question here</question>"
            )
            custom_user = f"Original: {clean_problem}\n\nReturn only the <question>...</question>."
            try:
                gen_backend = cfg.get("gen_backend", cfg.get("backend", "ollama"))
                raw = call_llm(gen_model, custom_system, custom_user,
                               backend=gen_backend,
                               think=False,
                               base_url=cfg.get("ollama_base_url", ""),
                               timeout=int(cfg.get("ollama_timeout", 0)))
                cleaned = clean_ollama_response(raw)
                if cleaned and is_valid_question(cleaned):
                    results[idx]["variant"] = cleaned
                    results[idx]["status"] = "testing"
                    yield make_yield()
                    finalize()
                    yield make_yield()
                    future = executor.submit(
                        _test_one_variation, target_model, cleaned, gt, answer_instruction, cfg)
                    pending_future = (future, idx)
                else:
                    results[idx]["status"] = "error"
                    results[idx]["error"] = "empty custom variation"
                    yield make_yield()
            except Exception as e:
                results[idx]["status"] = "error"
                results[idx]["error"] = str(e)
                yield make_yield()

    finalize()
    executor.shutdown(wait=False)
    yield make_yield()


def _stream_test_loop(results, baseline_html, baseline_correct, baseline_answer,
                       target_model, gt, answer_instruction, cfg):
    """Re-test existing variants (RETEST mode)."""
    executor = ThreadPoolExecutor(max_workers=1)
    pending_future = None

    def finalize():
        nonlocal pending_future
        if pending_future is None:
            return
        future, idx = pending_future
        try:
            drift_result = future.result(timeout=130)
            results[idx]["answer"] = drift_result["answer"]
            results[idx]["correct"] = drift_result["correct"]
            results[idx]["drifted"] = drift_result["correct"] != baseline_correct
            results[idx]["positive_drift"] = not baseline_correct and drift_result["correct"]
            results[idx]["negative_drift"] = baseline_correct and not drift_result["correct"]
            if drift_result.get("reasoning"):
                results[idx]["reasoning"] = drift_result["reasoning"]
            results[idx]["status"] = "done"
        except Exception as e:
            results[idx]["status"] = "error"
            results[idx]["error"] = str(e)
        pending_future = None

    def make_yield():
        return (
            baseline_html + render_cards(results),
            render_summary(results, baseline_correct, baseline_answer),
            render_drift_chart(results, baseline_correct),
            json.dumps([r for r in results if r.get("status") in ("done", "error")], indent=2),
        )

    yield make_yield()

    for i, r in enumerate(results):
        if r.get("status") != "re-testing":
            continue
        finalize()
        yield make_yield()
        future = executor.submit(
            _test_one_variation, target_model, r["variant"], gt, answer_instruction, cfg)
        pending_future = (future, i)

    finalize()
    executor.shutdown(wait=False)
    yield make_yield()
