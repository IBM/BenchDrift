"""
Novel feature discovery for BenchDrift pipeline.

Asks an LLM to identify structural features of a problem that are NOT
covered by the predefined feature set. Backend-agnostic via call_fn pattern.

Used by both the interactive app and potentially the batch pipeline.
"""

import json
import re
from typing import Callable, Dict, Optional, Tuple


def discover_novel_features(
    problem_text: str,
    existing_features: dict,
    call_fn: Optional[Callable[[str, str], str]] = None,
) -> Tuple[Dict[str, bool], str]:
    """Ask an LLM to identify novel features not in the predefined set.

    Args:
        problem_text: The problem to analyze.
        existing_features: Dict of already-known features (keys are feature names).
        call_fn: Optional callable(system_prompt, user_prompt) -> str.
            The caller provides this from whatever LLM backend they use.
            If None, returns empty dict.

    Returns:
        Tuple of (discovered_features_dict, status_message).
        discovered_features_dict: {feature_name: bool} for novel features.
        status_message: Human-readable status string.
    """
    if not problem_text or not problem_text.strip():
        return {}, ""
    if not call_fn:
        return {}, "No LLM backend provided"

    all_known = list(existing_features.keys())
    existing_list = ", ".join(all_known)

    system_prompt = (
        "You analyze math/reasoning problems for novel structural features. "
        "Return ONLY a JSON object with boolean values."
    )
    user_prompt = (
        f"You are analyzing a math/reasoning problem for NOVEL structural features "
        f"that are NOT covered by these existing features: {existing_list}.\n\n"
        f"Identify 1-5 NEW features specific to this problem that could affect how "
        f"an LLM processes it. Each feature should be:\n"
        f"- Named as has_<something> (snake_case)\n"
        f"- A boolean (true/false for this problem)\n"
        f"- Genuinely novel (not a synonym of existing features)\n\n"
        f"Problem: {problem_text.strip()[:500]}\n\n"
        f'Return ONLY a JSON object like '
        f'{{"has_recursive_structure": true, "has_visual_layout": false}}'
    )

    try:
        raw = call_fn(system_prompt, user_prompt)
    except Exception as e:
        return {}, f"LLM error: {e}"

    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return {}, "Could not parse LLM response"

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {}, "Invalid JSON from LLM"

    discovered = {}
    for k, v in parsed.items():
        if k.startswith("has_") and isinstance(v, bool) and k not in all_known:
            discovered[k] = v

    if not discovered:
        return {}, "No novel features discovered"
    return discovered, f"Discovered {len(discovered)} novel feature(s)"
