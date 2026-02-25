"""
Answer normalization and judge response parsing for BenchDrift evaluation.

Core utilities used by both the batch pipeline and the interactive app:
- normalize_for_judge(): Extract core answer from verbose ground truth
- parse_judge_response(): Parse YES/NO from LLM judge output
"""

import re
from typing import Optional


def normalize_for_judge(truth: str) -> str:
    """Extract core answer value from verbose ground truth.

    Handles common benchmark formats:
    - LaTeX \\boxed{...} (including nested braces)
    - GSM8K #### markers
    - "The final answer is X" patterns
    - Fallback: last number or last sentence

    Args:
        truth: Raw ground truth string from benchmark dataset.

    Returns:
        Normalized answer string suitable for judge comparison.
    """
    if not truth:
        return truth or ""

    # Extract from \boxed{...} — handles nested braces like \boxed{\frac{3}{4}}
    m = re.search(r'\\boxed\{', truth)
    if m:
        start = m.end()
        depth = 1
        i = start
        while i < len(truth) and depth > 0:
            if truth[i] == '{':
                depth += 1
            elif truth[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return truth[start:i - 1].strip()

    # GSM8K: #### answer
    m = re.search(r'####\s*(.+)', truth)
    if m:
        return m.group(1).strip()

    # Short answers: use as-is
    if len(truth) <= 200:
        return truth

    # "The (final) answer is X"
    m = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
                  truth, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: last number
    nums = re.findall(r'-?\d+\.?\d*', truth)
    if nums:
        return nums[-1]

    # Fallback: last sentence
    sentences = re.split(r'(?<=[.!?])\s+', truth.strip())
    if sentences:
        return sentences[-1].strip()

    return truth


def parse_judge_response(response: str) -> Optional[bool]:
    """Parse a YES/NO response from an LLM judge.

    Robust parsing that handles common LLM response patterns:
    - "YES" / "NO" (exact)
    - "Yes, because..." / "No, the answers differ..."
    - "YES." / "NO."
    - Avoids substring traps like "YESTERDAY" containing "YES"

    Args:
        response: Raw text response from the LLM judge.

    Returns:
        True for YES, False for NO, None if unparseable.
    """
    if not response or not response.strip():
        return None

    resp = response.strip().upper()
    first_word = resp.split()[0] if resp.split() else ""

    # Exact or first-word match
    if first_word == "YES" or resp == "YES":
        return True
    if first_word == "NO" or resp == "NO":
        return False

    # Starts-with (covers "YES." "YES," "NO." "NO,")
    if resp.startswith("YES"):
        return True
    if resp.startswith("NO"):
        return False

    return None
