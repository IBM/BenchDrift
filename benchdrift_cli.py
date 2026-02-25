#!/usr/bin/env python3
"""
BenchDrift CLI — Minimal command-line tool for single-problem variation generation.

This is the CLI counterpart of the Gradio app. Given a problem (and optionally
an expected answer), it generates ranked variations using the v2 pipeline
(feature analysis → relevance ranking → Ollama-based variation generation)
and optionally tests them against a target model.

Usage:
    # Generate variations only (no model needed for ranking)
    python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --top-k 5

    # Generate + test variations against a target model
    python benchdrift_cli.py \
        --problem "What is 15 + 25?" --answer "40" \
        --gen-model qwen3:8b --target-model mistral:7b --top-k 5

    # Read problem from stdin (pipe-friendly)
    echo "What is 15 + 25?" | python benchdrift_cli.py --answer "40" --top-k 3

    # Output as JSON for downstream processing
    python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --json

    # Include persona variations
    python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --use-axes "all"

    # Skip LLM enrichment (regex features only — instant)
    python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --no-enrich
"""

import argparse
import json
import os
import re
import sys
import time

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from benchdrift.pipeline.feature_relevance import (
    get_problem_features,
    enrich_features_with_llm,
    rank_transformations_two_level,
    _get_valid_axes,
    _rank_axes_by_features,
    parse_axes,
    TAXONOMY,
    TRANSFORMATION_TO_AXIS,
    WITHIN_AXIS_AFFINITY,
)
from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
from benchdrift.pipeline.comprehensive_variation_engine_v2 import (
    clean_model_response,
    is_valid_question,
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Valid client types (must match pipeline)
VALID_CLIENT_TYPES = {'rits', 'openai', 'vllm', 'vllm_logits', 'ollama', 'ollama_logits', 'groq'}


def parse_model_spec(spec: str, default_client: str = "ollama"):
    """Parse 'client/model' spec into (client_type, model_name).

    Examples:
        "ollama/qwen3:8b"   → ("ollama", "qwen3:8b")
        "groq/llama-3.3"    → ("groq", "llama-3.3")
        "qwen3:8b"          → ("ollama", "qwen3:8b")   # bare name → default
    """
    if '/' in spec:
        parts = spec.split('/', 1)
        candidate = parts[0].lower()
        if candidate in VALID_CLIENT_TYPES:
            return candidate, parts[1]
    return default_client, spec


# ---------------------------------------------------------------------------
# Model call helpers — routes to Ollama (lightweight) or model_client.py
# ---------------------------------------------------------------------------

# Cache for non-Ollama model clients (avoid re-creating on every call)
_model_client_cache = {}


def call_ollama(model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 1024, temperature: float = 0.5,
                base_url: str = "", timeout: int = 120) -> str:
    """Call Ollama /api/chat. Returns response text."""
    import requests
    url = base_url or OLLAMA_BASE_URL
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.85,
        },
        "think": False,
        "stream": False,
    }
    resp = requests.post(f"{url}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def _get_or_create_client(client_type: str, model_name: str):
    """Get or create a model client for non-Ollama backends."""
    cache_key = f"{client_type}:{model_name}"
    if cache_key not in _model_client_cache:
        from benchdrift.models.model_client import ModelClientFactory
        _model_client_cache[cache_key] = ModelClientFactory.create_client(
            client_type, model_name)
    return _model_client_cache[cache_key]


def call_model(client_type: str, model_name: str,
               system_prompt: str, user_prompt: str,
               max_tokens: int = 1024, temperature: float = 0.5,
               base_url: str = "", timeout: int = 120) -> str:
    """Unified model call — routes to Ollama or model_client.py based on client_type."""
    if client_type in ('ollama', 'ollama_logits'):
        return call_ollama(model_name, system_prompt, user_prompt,
                           max_tokens=max_tokens, temperature=temperature,
                           base_url=base_url, timeout=timeout)

    # Non-Ollama: use model_client.py
    client = _get_or_create_client(client_type, model_name)
    prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    responses = client.get_model_response([prompt])
    if responses and responses[0]:
        return responses[0].strip()
    return ""


def clean_ollama_response(raw: str) -> str:
    """Strip <think> tags, extract <question> tags, clean preamble."""
    if not raw:
        return ""
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = clean_model_response(text)
    return text


def generate_one_variation(client_type: str, model: str, problem: str,
                           trans_name: str, config: dict,
                           base_url: str = "", timeout: int = 120,
                           max_retries: int = 2) -> str:
    """Generate a single variation. Routes to Ollama or model_client based on client_type.
    Returns variation text, 'SKIP', or ''."""
    system_prompt = (
        f"You are an expert at creating intent-preserving question variations.\n\n"
        f"TASK: Create a {trans_name} variation of the given problem.\n\n"
        f"TRANSFORMATION GOAL: {config['prompt']}\n\n"
        f"UNIVERSAL RULES:\n"
        f"1. PRESERVE the exact answer\n"
        f"2. MAINTAIN all mathematical/logical relationships\n"
        f"3. Numbers: format can change (5 -> five), value CANNOT (5 -> 6)\n"
        f"4. Units: convert correctly or not at all\n"
        f"5. Use PLAIN TEXT only\n"
        f"6. Return ONLY the question inside <question> tags\n"
        f"7. Do NOT explain your reasoning\n\n"
        f"<question>Your transformed question here</question>"
    )
    user_prompt = f"Original: {problem}\n\nReturn only the <question>...</question>."

    for attempt in range(max_retries + 1):
        try:
            raw = call_model(client_type, model, system_prompt, user_prompt,
                             max_tokens=1024, temperature=0.5,
                             base_url=base_url, timeout=timeout)
        except Exception as e:
            if attempt == max_retries:
                return ""
            continue

        cleaned = clean_ollama_response(raw)
        if cleaned and cleaned.strip().upper() == "SKIP":
            return "SKIP"
        if cleaned and is_valid_question(cleaned):
            return cleaned

        # Retry with stricter prompt
        user_prompt = (
            f"Original: {problem}\n\n"
            f"Return ONLY the question text inside <question> tags. "
            f"No explanation, no analysis."
        )

    return ""


def answers_match(predicted: str, truth: str) -> bool:
    """Robust answer comparison: numeric, fraction, and text."""
    def _normalize(s: str) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        for prefix in ("the answer is", "answer:", "solution:", "result:", "final answer:"):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        for suffix in (".", "!", "?"):
            if s.endswith(suffix):
                s = s[:-1].strip()
        frac = re.search(r'-?\d+\s*/\s*\d+', s)
        if frac:
            return re.sub(r'\s', '', frac.group())
        s_nc = re.sub(r'(\d),(\d)', r'\1\2', s)
        nums = re.findall(r'-?\d+\.?\d*', s_nc)
        if nums:
            return nums[0]
        return s

    a, b = _normalize(predicted), _normalize(truth)
    if a == b:
        return True
    try:
        def _to_float(x):
            if '/' in x:
                num, den = x.split('/')
                return float(num) / float(den)
            return float(x)
        return abs(_to_float(a) - _to_float(b)) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass
    return a in b or b in a


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BenchDrift CLI — generate and test problem variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick analysis + variations (regex features only, no LLM enrichment)
  python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --no-enrich

  # Full pipeline with LLM enrichment and target model testing
  python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" \\
      --gen-model qwen3:8b --target-model mistral:7b --top-k 5

  # JSON output for scripting
  python benchdrift_cli.py --problem "What is 15 + 25?" --answer "40" --json

  # Read problem from stdin
  echo "What is 15 + 25?" | python benchdrift_cli.py --answer "40"
""")

    # Input
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem text (reads from stdin if not provided)")
    parser.add_argument("--answer", type=str, default=None,
                        help="Expected answer (required for drift testing)")

    # Models (accept "client/model" format, e.g., ollama/qwen3:8b)
    parser.add_argument("--gen-model", type=str, default="qwen3:8b",
                        help="Model for generating variations. Accepts 'client/model' format "
                             "(default: qwen3:8b, uses Ollama)")
    parser.add_argument("--target-model", type=str, default=None,
                        help="Model to test variations against. Accepts 'client/model' format "
                             "(skip testing if not set)")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_BASE_URL,
                        help=f"Ollama base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Ollama timeout in seconds (default: 120)")

    # Variation control
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top-ranked variations to generate (default: 10)")
    parser.add_argument("--use-axes", type=str,
                        default="linguistic,referential,pragmatic,structural,constraint_targeted",
                        help="Comma-separated taxonomy axes to enable. "
                             "Valid: linguistic,referential,pragmatic,structural,persona,long_context,constraint_targeted,all. "
                             "Subtract with minus: 'all,-persona'. (default: all non-persona/non-long_context)")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip LLM feature enrichment (regex only — instant)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Only show feature analysis and ranking — don't generate variations")

    # Output
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to file")

    args = parser.parse_args()

    # ── Resolve client/model specs ──
    args._gen_client, args.gen_model = parse_model_spec(args.gen_model, "ollama")
    if args.target_model:
        args._target_client, args.target_model = parse_model_spec(args.target_model, "ollama")
    else:
        args._target_client = "ollama"

    # Get problem text
    problem = args.problem
    if problem is None:
        if sys.stdin.isatty():
            parser.error("--problem is required (or pipe via stdin)")
        problem = sys.stdin.read().strip()
    if not problem:
        parser.error("Empty problem text")

    # ── Phase 1: Feature Analysis ──
    if not args.json:
        print("=" * 70)
        print("BENCHDRIFT — Variation Analysis & Generation")
        print("=" * 70)
        print(f"\nProblem: {problem[:200]}{'...' if len(problem) > 200 else ''}")
        if args.answer:
            print(f"Answer:  {args.answer}")
        print()

    features = get_problem_features(problem)

    # LLM enrichment (optional)
    if not args.no_enrich:
        if not args.json:
            print("Enriching features via LLM...", end=" ", flush=True)
        try:
            llm_feats = enrich_features_with_llm(
                problem, args.ollama_url, args.gen_model, timeout=args.timeout)
            features.update(llm_feats)
            if not args.json:
                print(f"done ({len(llm_feats)} LLM features)")
        except Exception as e:
            if not args.json:
                print(f"skipped ({e})")

    # Show active features
    active = [k for k, v in features.items()
              if v and k not in ('num_numbers', 'num_sentences', 'problem_length')]
    if not args.json:
        print(f"\nActive features ({len(active)}):")
        print(f"  {', '.join(sorted(active))}")

    # ── Phase 2: Relevance Ranking ──
    enabled_axes = parse_axes(args.use_axes)
    all_types = UnifiedVariationEngine.get_all_transformation_types()

    # Filter transformation types to only those in enabled axes
    all_types = {k: v for k, v in all_types.items()
                 if TRANSFORMATION_TO_AXIS.get(k) in enabled_axes}

    valid_axes = _get_valid_axes(features, enabled_axes=enabled_axes)
    ranked_axes = _rank_axes_by_features(features, valid_axes)

    ranked = rank_transformations_two_level(
        problem, features, all_types,
        top_k=args.top_k,
        pre_ranked_axes=ranked_axes,
        enabled_axes=enabled_axes,
    )

    if not args.json:
        print(f"\nAxis ranking: {' > '.join(ranked_axes)}")
        print(f"\nTop-{args.top_k} transformations:")
        for i, (name, score, axis) in enumerate(ranked, 1):
            print(f"  {i:2d}. {name:40s} {score:.3f}  [{axis}]")

    # ── Phase 3: Baseline test (run original problem through target model) ──
    baseline_result = None
    if args.target_model and args.answer and not args.no_generate:
        if not args.json:
            print(f"\nTesting baseline on {args.target_model}...", end=" ", flush=True)
        try:
            t0 = time.time()
            raw = call_model(
                args._target_client, args.target_model,
                "Solve the problem. Return ONLY the final answer. No explanation.",
                problem,
                max_tokens=256, temperature=0.0,
                base_url=args.ollama_url, timeout=args.timeout,
            )
            predicted = re.sub(r'<think>.*?</think>', '', raw,
                               flags=re.DOTALL | re.IGNORECASE).strip()
            baseline_correct = answers_match(predicted, args.answer)
            elapsed = time.time() - t0
            baseline_result = {
                "answer": predicted,
                "correct": baseline_correct,
                "time_s": round(elapsed, 1),
            }
            if not args.json:
                status_str = "CORRECT" if baseline_correct else "WRONG"
                print(f"{status_str} ({elapsed:.1f}s)  answer={predicted[:60]}")
        except Exception as e:
            baseline_result = {"answer": f"ERROR: {e}", "correct": None, "time_s": 0}
            if not args.json:
                print(f"ERROR: {e}")

    # ── Phase 4: Generate Variations ──
    results = []
    if not args.no_generate:
        if not args.json:
            print(f"\nGenerating {len(ranked)} variations via {args.gen_model}...")
            print("-" * 70)

        for i, (trans_name, score, axis) in enumerate(ranked, 1):
            config = all_types[trans_name]
            if not args.json:
                print(f"  [{i}/{len(ranked)}] {trans_name}...", end=" ", flush=True)

            t0 = time.time()
            variation = generate_one_variation(
                args._gen_client, args.gen_model, problem, trans_name, config,
                base_url=args.ollama_url, timeout=args.timeout,
            )
            elapsed = time.time() - t0

            entry = {
                "transformation": trans_name,
                "axis": axis,
                "relevance_score": score,
                "variation": variation,
                "status": "skip" if variation == "SKIP" else ("ok" if variation else "fail"),
                "generation_time_s": round(elapsed, 1),
            }

            # ── Phase 5: Test variation against target model ──
            if args.target_model and args.answer and variation and variation not in ("SKIP", ""):
                try:
                    raw = call_model(
                        args._target_client, args.target_model,
                        "Solve the problem. Return ONLY the final answer. No explanation.",
                        variation,
                        max_tokens=256, temperature=0.0,
                        base_url=args.ollama_url, timeout=args.timeout,
                    )
                    predicted = re.sub(r'<think>.*?</think>', '', raw,
                                       flags=re.DOTALL | re.IGNORECASE).strip()
                    correct = answers_match(predicted, args.answer)
                    entry["target_answer"] = predicted
                    entry["correct"] = correct

                    # Compute drift type relative to baseline
                    if baseline_result and baseline_result.get("correct") is not None:
                        if baseline_result["correct"] and not correct:
                            entry["drift_type"] = "negative"  # was right, now wrong
                        elif not baseline_result["correct"] and correct:
                            entry["drift_type"] = "positive"  # was wrong, now right
                        elif baseline_result["correct"] and correct:
                            entry["drift_type"] = "none"      # both correct
                        else:
                            entry["drift_type"] = "both_wrong" # neither correct
                except Exception as e:
                    entry["target_answer"] = f"ERROR: {e}"
                    entry["correct"] = None

            results.append(entry)

            if not args.json:
                status = entry["status"]
                if status == "skip":
                    print(f"SKIP ({elapsed:.1f}s)")
                elif status == "fail":
                    print(f"FAIL ({elapsed:.1f}s)")
                else:
                    preview = variation[:80] + ("..." if len(variation) > 80 else "")
                    drift_label = ""
                    if "drift_type" in entry:
                        dt = entry["drift_type"]
                        if dt == "negative":
                            drift_label = " NEG-DRIFT"
                        elif dt == "positive":
                            drift_label = " POS-DRIFT"
                        elif dt == "none":
                            drift_label = " OK"
                        elif dt == "both_wrong":
                            drift_label = " BOTH-WRONG"
                    elif "correct" in entry:
                        drift_label = " CORRECT" if entry["correct"] else " WRONG"
                    print(f"OK ({elapsed:.1f}s){drift_label}")
                    print(f"    {preview}")

    # ── Summary ──
    output = {
        "problem": problem,
        "answer": args.answer,
        "features": {k: v for k, v in features.items()
                     if k not in ('num_numbers', 'num_sentences', 'problem_length')},
        "axis_ranking": ranked_axes,
        "ranked_transformations": [
            {"name": n, "score": s, "axis": a} for n, s, a in ranked
        ],
        "baseline": baseline_result,
        "variations": results,
    }

    if results and args.target_model:
        tested = [r for r in results if r.get("correct") is not None]
        neg_drift = sum(1 for r in tested if r.get("drift_type") == "negative")
        pos_drift = sum(1 for r in tested if r.get("drift_type") == "positive")
        both_correct = sum(1 for r in tested if r.get("drift_type") == "none")
        both_wrong = sum(1 for r in tested if r.get("drift_type") == "both_wrong")
        # Fallback for cases without baseline
        correct_count = sum(1 for r in tested if r["correct"])

        output["drift_summary"] = {
            "tested": len(tested),
            "baseline_correct": baseline_result["correct"] if baseline_result else None,
            "negative_drift": neg_drift,
            "positive_drift": pos_drift,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "variation_accuracy": round(correct_count / len(tested), 3) if tested else 0,
            "drift_rate": round(neg_drift / len(tested), 3) if tested else 0,
        }
        if not args.json:
            print(f"\n{'=' * 70}")
            bl_str = "CORRECT" if baseline_result and baseline_result.get("correct") else "WRONG"
            print(f"BASELINE: {bl_str}  (answer: {baseline_result['answer'][:60] if baseline_result else 'N/A'})")
            print(f"DRIFT SUMMARY ({len(tested)} tested):")
            print(f"  Negative drift (was right, now wrong): {neg_drift}")
            print(f"  Positive drift (was wrong, now right): {pos_drift}")
            print(f"  Both correct:                          {both_correct}")
            print(f"  Both wrong:                            {both_wrong}")
            print(f"  Drift rate: {output['drift_summary']['drift_rate']:.1%}")
            print(f"{'=' * 70}")

    # JSON output
    if args.json:
        print(json.dumps(output, indent=2, default=str))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        if not args.json:
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
