"""
Run strategies and metadata tracking for BenchDrift pipeline.

Defines run modes (FULL, RETEST, REEVAL, INCREMENTAL) and provides
metadata utilities for smart re-run detection — determining the cheapest
execution path when configuration changes.

Used by both the batch pipeline and the interactive app.
"""

import hashlib
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RunMode(Enum):
    """Pipeline run strategies."""
    FULL = "full"
    RETEST = "retest"
    REEVAL = "reeval"
    INCREMENTAL = "incremental"


def build_meta(problem: str, gen_model: str, target_model: str,
               cfg: dict, strip_instruction_fn=None) -> dict:
    """Build metadata dict for a pipeline run.

    Args:
        problem: The problem text.
        gen_model: Generator model name.
        target_model: Target model name.
        cfg: Pipeline configuration dict.
        strip_instruction_fn: Optional callable(str) -> (str, str) to strip
            instruction tags from problem text. If None, uses problem as-is.

    Returns:
        Metadata dict with problem hash and all relevant config values.
    """
    if strip_instruction_fn and problem:
        clean, _ = strip_instruction_fn(problem.strip())
    else:
        clean = problem.strip() if problem else ""

    return {
        "_run_meta": True,
        "problem_hash": hashlib.sha256(clean.strip().encode()).hexdigest()[:16],
        "gen_model": gen_model or "",
        "target_model": target_model or "",
        "variation_mode": cfg.get("variation_mode", "axes-based"),
        "eval_method": cfg.get("eval_method", "string matching"),
        "judge_model": cfg.get("judge_model", ""),
        "validate_variations": cfg.get("validate_variations", True),
        "gen_temperature": cfg.get("gen_temperature", 0.5),
        "gen_max_tokens": cfg.get("gen_max_tokens", 1024),
        "gen_max_retries": cfg.get("gen_max_retries", 2),
        "solver_temperature": cfg.get("solver_temperature", 0.0),
        "solver_max_tokens": cfg.get("solver_max_tokens", 256),
    }


def parse_prev(prev_json: str) -> Tuple[list, Optional[dict]]:
    """Parse previous results JSON into (results, metadata).

    Args:
        prev_json: JSON string of previous run results.

    Returns:
        Tuple of (results_list, metadata_dict_or_None).
    """
    try:
        data = json.loads(prev_json) if prev_json else []
        if not isinstance(data, list):
            return [], None
    except (json.JSONDecodeError, TypeError):
        return [], None

    meta = None
    results = []
    for item in data:
        if isinstance(item, dict) and item.get("_run_meta"):
            meta = item
        else:
            results.append(item)
    return results, meta


def inject_meta(results_json: str, meta: dict) -> str:
    """Inject metadata into results JSON.

    Args:
        results_json: JSON string of results.
        meta: Metadata dict to inject.

    Returns:
        JSON string with metadata prepended.
    """
    try:
        data = json.loads(results_json) if results_json else []
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, TypeError):
        data = []
    data = [d for d in data if not (isinstance(d, dict) and d.get("_run_meta"))]
    return json.dumps([meta] + data, indent=2)


def generation_changed(prev_meta: dict, current_meta: dict) -> bool:
    """Check if generation parameters changed between runs."""
    keys = ("gen_model", "gen_temperature", "gen_max_tokens",
            "gen_max_retries", "validate_variations", "variation_mode")
    return any(prev_meta.get(k) != current_meta.get(k) for k in keys)


def testing_changed(prev_meta: dict, current_meta: dict) -> bool:
    """Check if testing parameters changed between runs."""
    keys = ("target_model", "solver_temperature", "solver_max_tokens")
    return any(prev_meta.get(k) != current_meta.get(k) for k in keys)


def eval_changed(prev_meta: dict, current_meta: dict) -> bool:
    """Check if evaluation parameters changed between runs."""
    keys = ("eval_method", "judge_model")
    return any(prev_meta.get(k) != current_meta.get(k) for k in keys)


def detect_run_mode(prev_meta: Optional[dict], current_meta: dict,
                    has_prev_results: bool) -> RunMode:
    """Determine the cheapest run mode given previous and current config.

    Args:
        prev_meta: Metadata from previous run (None if no previous run).
        current_meta: Metadata for current run.
        has_prev_results: Whether previous results exist.

    Returns:
        The appropriate RunMode.
    """
    if not has_prev_results or not prev_meta:
        return RunMode.FULL
    if generation_changed(prev_meta, current_meta):
        return RunMode.FULL
    if testing_changed(prev_meta, current_meta):
        return RunMode.RETEST
    if eval_changed(prev_meta, current_meta):
        return RunMode.REEVAL
    return RunMode.INCREMENTAL
