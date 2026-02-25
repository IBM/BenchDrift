"""
Judge Few-Shot Examples — trait-based composition for LLM judge.

Instead of one category → one block, we detect MULTIPLE traits from the expected
answer and compose a few-shot block from all matching traits. This handles
overlapping answer styles naturally (e.g. "Compound 3: H2SO4" triggers both
`labeled_choice` and `chemical_formula` traits).

Detection is zero-cost: regex on expected answer + optional HF dataset name.

Optional LLM fallback (off by default, enable via cfg["llm_trait_fallback"]):
When regex detects zero specific traits, a single cheap LLM call classifies
the answer into existing traits. Results are cached at the pattern level
so the same answer style only triggers one LLM call ever.
"""

import logging
import re
import threading
from typing import Dict, List, Optional, Set

logger = logging.getLogger("BenchDrift")

# ──────────────────────────────────────────────────────────────────────────────
# TRAIT BANK — each trait contributes 2-4 YES/NO examples
# ──────────────────────────────────────────────────────────────────────────────
# Traits represent HOW to compare, not what domain the answer is from.
# Multiple traits can fire for a single answer → examples are composed.

TRAITS = {
    # ── Label / option identity ──
    "labeled_choice": {
        "detect": "regex",  # detected by regex on answer
        "examples": [
            'GT: "Mutant 2" vs PRED: "Mutant 1: 5\'-ATGTTCTACGCT..." → NO (label 1 ≠ label 2)',
            'GT: "Mutant 2" vs PRED: "Mutant 2: 5\'-ATGTTCTAAGCT..." → YES (same label, extra detail is fine)',
            'GT: "Mutant 2" vs PRED: "Version 2" → YES (same item #2, different naming)',
            'GT: "Compound 3" vs PRED: "Compound 1" → NO (different label numbers)',
        ],
        "preamble": "CRITICAL: The label/identifier IS the answer. Only the label must match — extra detail is irrelevant.",
    },

    "mcq_letter": {
        "detect": "regex",
        "examples": [
            'GT: "A" vs PRED: "A. Mitosis" → YES (same letter)',
            'GT: "A" vs PRED: "B" → NO (different letters)',
            'GT: "B" vs PRED: "(B)" → YES (same letter, formatting differs)',
            'GT: "C" vs PRED: "The answer is C because..." → YES (same letter)',
        ],
        "preamble": "CRITICAL: The answer is a multiple-choice letter. ONLY the letter matters.",
    },

    "roman_numeral": {
        "detect": "regex",
        "examples": [
            'GT: "II" vs PRED: "Statement II" → YES (same numeral)',
            'GT: "II" vs PRED: "III" → NO (different numerals)',
            'GT: "IV" vs PRED: "4" → YES (same value, roman vs arabic)',
        ],
        "preamble": "The answer uses Roman numerals. Match the numeral value.",
    },

    # ── Numeric ──
    "numeric_integer": {
        "detect": "regex",
        "examples": [
            'GT: "42" vs PRED: "42.0" → YES (same value)',
            'GT: "42" vs PRED: "The answer is 42" → YES (extract the number)',
            'GT: "42" vs PRED: "43" → NO (different values)',
            'GT: "1000" vs PRED: "1,000" → YES (comma formatting)',
        ],
        "preamble": "The answer is a number. Compare numerical VALUE, ignoring format.",
    },

    "numeric_decimal": {
        "detect": "regex",
        "examples": [
            'GT: "3.14" vs PRED: "3.14159" → NO (different precision = different value unless context says round)',
            'GT: "0.5" vs PRED: "1/2" → YES (same value)',
            'GT: "15.5" vs PRED: "fifteen and a half" → YES (same value in words)',
        ],
        "preamble": "The answer is a decimal number. Compare the numerical value.",
    },

    "numeric_fraction": {
        "detect": "regex",
        "examples": [
            'GT: "3/4" vs PRED: "0.75" → YES (same value)',
            'GT: "3/4" vs PRED: "4/3" → NO (different fractions)',
            'GT: "1/2" vs PRED: "half" → YES (same value in words)',
        ],
        "preamble": "The answer is a fraction. Compare the numerical value it represents.",
    },

    "numeric_percentage": {
        "detect": "regex",
        "examples": [
            'GT: "25%" vs PRED: "0.25" → YES (same value)',
            'GT: "25%" vs PRED: "25 percent" → YES (same value)',
            'GT: "25%" vs PRED: "75%" → NO (different percentages)',
        ],
        "preamble": "The answer is a percentage. Compare the value.",
    },

    "numeric_negative": {
        "detect": "regex",
        "examples": [
            'GT: "-7" vs PRED: "7" → NO (sign matters)',
            'GT: "-3.5" vs PRED: "-3.5" → YES',
            'GT: "-7" vs PRED: "negative seven" → YES (same value)',
        ],
        "preamble": "The answer includes a negative number. The SIGN is critical.",
    },

    # ── Math / symbolic expressions ──
    "expression_latex": {
        "detect": "regex",
        "examples": [
            'GT: "\\frac{3}{4}" vs PRED: "3/4" → YES (same expression)',
            'GT: "\\sqrt{2}" vs PRED: "sqrt(2)" → YES (different notation)',
            'GT: "\\frac{3}{4}" vs PRED: "\\frac{4}{3}" → NO (different fractions)',
            'GT: "\\boxed{42}" vs PRED: "42" → YES (boxed formatting)',
        ],
        "preamble": "The answer is a LaTeX expression. Compare mathematical MEANING, not formatting.",
    },

    "expression_algebraic": {
        "detect": "regex",
        "examples": [
            'GT: "x^2 + 1" vs PRED: "1 + x^2" → YES (same expression, reordered)',
            'GT: "2x + 3" vs PRED: "3 + 2x" → YES (same polynomial)',
            'GT: "x^2 + 1" vs PRED: "x^2 - 1" → NO (different expressions)',
            'GT: "2\\pi" vs PRED: "2*pi" → YES (same value)',
        ],
        "preamble": "The answer is an algebraic expression. Compare the mathematical meaning.",
    },

    # ── Scientific notation / formulas ──
    "chemical_formula": {
        "detect": "regex",
        "examples": [
            'GT: "H2SO4" vs PRED: "sulfuric acid" → YES (same compound)',
            'GT: "NaCl" vs PRED: "sodium chloride" → YES (same compound)',
            'GT: "H2O" vs PRED: "H2O2" → NO (different compounds — water vs hydrogen peroxide)',
            'GT: "CH4" vs PRED: "methane" → YES (same molecule)',
        ],
        "preamble": "The answer is a chemical formula/compound. Match the chemical identity.",
    },

    "unit_quantity": {
        "detect": "regex",
        "examples": [
            'GT: "9.8 m/s²" vs PRED: "9.8" → YES (same value, unit omitted)',
            'GT: "100 cm" vs PRED: "1 m" → YES (same length, different units)',
            'GT: "273 K" vs PRED: "0°C" → YES (same temperature)',
            'GT: "5 kg" vs PRED: "5000 g" → YES (same mass)',
            'GT: "9.8 m/s²" vs PRED: "10 m/s²" → NO (different values)',
        ],
        "preamble": "The answer includes units. Match the QUANTITY — equivalent unit conversions are OK.",
    },

    "scientific_notation": {
        "detect": "regex",
        "examples": [
            'GT: "3.0 × 10^8" vs PRED: "3e8" → YES (same value)',
            'GT: "6.022 × 10^23" vs PRED: "Avogadro\'s number" → YES (same value)',
            'GT: "1.6 × 10^-19" vs PRED: "1.6e-19" → YES (same value)',
        ],
        "preamble": "The answer is in scientific notation. Compare the numerical value.",
    },

    # ── Date / time ──
    "date_time": {
        "detect": "regex",
        "examples": [
            'GT: "March 15, 44 BC" vs PRED: "15 March 44 BC" → YES (same date)',
            'GT: "1776" vs PRED: "The year 1776" → YES (same year)',
            'GT: "14:30" vs PRED: "2:30 PM" → YES (same time)',
            'GT: "6:55 AM" vs PRED: "06:55" → YES (same time)',
            'GT: "Monday" vs PRED: "Tuesday" → NO (different days)',
        ],
        "preamble": "The answer involves a date or time. Match the temporal value.",
    },

    # ── Boolean / binary ──
    "boolean": {
        "detect": "regex",
        "examples": [
            'GT: "True" vs PRED: "true" → YES (case difference)',
            'GT: "True" vs PRED: "Yes" → YES (both affirmative)',
            'GT: "True" vs PRED: "False" → NO (opposite)',
            'GT: "Yes" vs PRED: "No" → NO (opposite)',
        ],
        "preamble": "The answer is binary (yes/no, true/false). Only the truth value matters.",
    },

    # ── Categorical / word-phrase ──
    "categorical_concept": {
        "detect": "fallback",  # fires when nothing else matches
        "examples": [
            'GT: "mitosis" vs PRED: "Mitosis" → YES (case difference)',
            'GT: "photosynthesis" vs PRED: "the process of photosynthesis" → YES (same concept)',
            'GT: "evaporation" vs PRED: "condensation" → NO (opposite processes)',
        ],
        "preamble": "The answer is a concept/term. Match the meaning, not exact wording.",
    },

    "categorical_entity": {
        "detect": "regex",
        "examples": [
            'GT: "France" vs PRED: "Republic of France" → YES (same country)',
            'GT: "Paris" vs PRED: "London" → NO (different cities)',
            'GT: "Einstein" vs PRED: "Albert Einstein" → YES (same person)',
            'GT: "DNA" vs PRED: "deoxyribonucleic acid" → YES (same molecule)',
        ],
        "preamble": "The answer is a named entity. Abbreviations, full names, and common aliases are equivalent.",
    },

    "categorical_direction": {
        "detect": "regex",
        "examples": [
            'GT: "north" vs PRED: "northward" → YES (same direction)',
            'GT: "north" vs PRED: "south" → NO (opposite directions)',
            'GT: "northeast" vs PRED: "NE" → YES (same direction, abbreviated)',
            'GT: "left" vs PRED: "go left" → YES (same direction)',
        ],
        "preamble": "The answer is a direction. Match the direction value.",
    },

    # ── Sequences / ordered ──
    "sequence_list": {
        "detect": "regex",
        "examples": [
            'GT: "A, B, C" vs PRED: "A, B, C" → YES (same order)',
            'GT: "A, B, C" vs PRED: "A, C, B" → NO (different order matters)',
            'GT: "[1, 2, 3]" vs PRED: "1, 2, 3" → YES (formatting differs)',
            'GT: "1, 2, 3" vs PRED: "1, 2, 4" → NO (different elements)',
        ],
        "preamble": "The answer is a list/sequence. ORDER MATTERS unless stated otherwise.",
    },

    "ordering_ranking": {
        "detect": "regex",
        "examples": [
            'GT: "B > A > C" vs PRED: "B, A, C" → YES (same ranking)',
            'GT: "A > B > C" vs PRED: "C > B > A" → NO (reversed order)',
            'GT: "1st: X, 2nd: Y" vs PRED: "X, Y" → YES (same ranking)',
        ],
        "preamble": "The answer is a ranking/ordering. The ORDER is the answer.",
    },

    # ── Multi-part / structured ──
    "multi_part": {
        "detect": "regex",
        "examples": [
            'GT: "(3, 5)" vs PRED: "x=3, y=5" → YES (same values)',
            'GT: "(3, 5)" vs PRED: "(5, 3)" → NO (order matters in tuples)',
            'GT: "x=2, y=3" vs PRED: "2 and 3" → YES (same values, if order matches)',
        ],
        "preamble": "The answer has multiple parts. ALL parts must match.",
    },

    "json_structured": {
        "detect": "regex",
        "examples": [
            'GT: "42" vs PRED: \'{"answer": "42"}\' → YES (extract answer field)',
            'GT: "42" vs PRED: \'{"explanation": "...", "answer": "42"}\' → YES (ignore explanation)',
            'GT: "[A, B]" vs PRED: \'{"orderedlist": ["A", "B"]}\' → YES (same list)',
        ],
        "preamble": "If the answer contains JSON, extract the 'answer' field. Ignore explanation/reasoning fields.",
    },

    # ── Code ──
    "code_output": {
        "detect": "regex",
        "examples": [
            'GT: "Hello World" vs PRED: "Hello World\\n" → YES (trailing newline)',
            'GT: "[1, 4, 9]" vs PRED: "[1,4,9]" → YES (spacing differs)',
            'GT: "True" vs PRED: "true" → YES (language-specific boolean casing)',
            'GT: "3.0" vs PRED: "3" → NO (in code output, types may matter — check context)',
        ],
        "preamble": "The answer is a code output. Match the output value, ignoring trailing whitespace.",
    },

    # ── Range / interval ──
    "range_interval": {
        "detect": "regex",
        "examples": [
            'GT: "[2, 7]" vs PRED: "between 2 and 7" → YES (same range)',
            'GT: "(0, 1)" vs PRED: "0 to 1 exclusive" → YES (same interval)',
            'GT: "[2, 7]" vs PRED: "[2, 8]" → NO (different upper bound)',
        ],
        "preamble": "The answer is a range/interval. Match the bounds and inclusivity.",
    },

    # ── Coordinate / spatial ──
    "coordinate": {
        "detect": "regex",
        "examples": [
            'GT: "(3, 4)" vs PRED: "x=3, y=4" → YES (same point)',
            'GT: "(3, 4)" vs PRED: "(4, 3)" → NO (different point — order matters)',
            'GT: "row 2, column 5" vs PRED: "(2, 5)" → YES (same position)',
        ],
        "preamble": "The answer is a coordinate/position. Both components must match in order.",
    },

    # ── Ratio / proportion ──
    "ratio": {
        "detect": "regex",
        "examples": [
            'GT: "3:1" vs PRED: "3 to 1" → YES (same ratio)',
            'GT: "3:1" vs PRED: "1:3" → NO (inverted ratio)',
            'GT: "2:3:5" vs PRED: "2 to 3 to 5" → YES (same ratio)',
        ],
        "preamble": "The answer is a ratio. Match the ratio value and order.",
    },

    # ── Molecular biology / sequences ──
    "bio_sequence": {
        "detect": "regex",
        "examples": [
            'GT: "5\'-ATGCGT-3\'" vs PRED: "ATGCGT" → YES (same sequence, notation differs)',
            'GT: "5\'-ATGCGT-3\'" vs PRED: "5\'-ATGCGA-3\'" → NO (last base differs)',
            'GT: "Ala-Gly-Pro" vs PRED: "AGP" → YES (same peptide, different notation)',
        ],
        "preamble": "The answer is a biological sequence. Match the actual sequence content.",
    },

    # ── Explanation-bearing (extract conclusion) ──
    "answer_with_explanation": {
        "detect": "regex",
        "examples": [
            'GT: "42" vs PRED: "The answer is 42 because 6*7=42" → YES (extract answer, ignore reasoning)',
            'GT: "Paris" vs PRED: "Paris, since it is the capital of France" → YES (extract answer)',
            'GT: "No" vs PRED: "No, this is incorrect because..." → YES (extract the No)',
        ],
        "preamble": "The predicted answer may include explanations. EXTRACT the core answer first, THEN compare.",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# HF dataset → trait hints (can suggest multiple traits)
# ──────────────────────────────────────────────────────────────────────────────

_DATASET_TRAIT_HINTS = {
    # Math: numeric + expression
    "gsm8k": ["numeric_integer", "answer_with_explanation"],
    "openai/gsm8k": ["numeric_integer", "answer_with_explanation"],
    "math": ["expression_latex", "numeric_integer", "numeric_fraction"],
    "hendrycks/math": ["expression_latex", "numeric_integer", "numeric_fraction"],
    "lighteval/MATH-Hard": ["expression_latex", "expression_algebraic", "numeric_integer"],
    "math-hard": ["expression_latex", "expression_algebraic", "numeric_integer"],
    "competition_math": ["expression_latex", "numeric_integer"],
    "aqua_rat": ["numeric_integer", "mcq_letter"],
    "asdiv": ["numeric_integer"],
    "svamp": ["numeric_integer"],
    "mawps": ["numeric_integer"],

    # MCQ
    "mmlu": ["mcq_letter"],
    "cais/mmlu": ["mcq_letter"],
    "hails/mmlu_no_train": ["mcq_letter"],
    "lukaemon/mmlu": ["mcq_letter"],
    "tasksource/mmlu": ["mcq_letter"],
    "arc": ["mcq_letter"],
    "ai2_arc": ["mcq_letter"],
    "allenai/ai2_arc": ["mcq_letter"],
    "hellaswag": ["mcq_letter"],
    "winogrande": ["mcq_letter"],
    "piqa": ["mcq_letter"],
    "social_iqa": ["mcq_letter"],
    "commonsense_qa": ["mcq_letter"],
    "openbookqa": ["mcq_letter"],
    "sciq": ["mcq_letter"],
    "medmcqa": ["mcq_letter"],
    "medqa": ["mcq_letter"],

    # Science — labeled choice + possible formulas
    "gpqa": ["labeled_choice", "chemical_formula", "bio_sequence"],
    "Idavidrein/gpqa": ["labeled_choice", "chemical_formula", "bio_sequence"],
    "gpqa_diamond": ["labeled_choice", "chemical_formula", "bio_sequence"],

    # Boolean
    "boolq": ["boolean"],
    "super_glue/boolq": ["boolean"],

    # QA / fill-in-blank
    "triviaqa": ["categorical_entity", "answer_with_explanation"],
    "nq_open": ["categorical_entity"],
    "web_questions": ["categorical_entity"],
    "squad": ["categorical_entity", "answer_with_explanation"],
    "squad_v2": ["categorical_entity", "answer_with_explanation"],
    "natural_questions": ["categorical_entity"],

    # Truthful QA
    "truthful_qa": ["categorical_concept", "answer_with_explanation"],

    # Code
    "humaneval": ["code_output"],
    "openai_humaneval": ["code_output"],
    "mbpp": ["code_output"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Trait detection — regex on expected answer, zero LLM cost
# ──────────────────────────────────────────────────────────────────────────────

# Patterns for labeled choices
_LABELED_WORDS = (
    r'(?i)^(mutant|option|choice|compound|statement|version|variant|'
    r'reaction|solution|formula|equation|sequence|sample|'
    r'experiment|group|strain|allele|genotype|phenotype|'
    r'hypothesis|theory|model|method|approach|technique|'
    r'figure|table|case|scenario|condition|treatment|'
    r'item|type|class|category|set|pair|configuration|'
    r'diagram|graph|pathway|mechanism|process|step)\s'
)

# Chemical formula: H2O, NaCl, C6H12O6, CH3COOH, Fe2O3
_CHEMICAL_RE = re.compile(
    r'^[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?){1,}$'
)

# Direction words
_DIRECTIONS = {
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "left", "right",
    "northward", "southward", "eastward", "westward",
    "clockwise", "counterclockwise", "ne", "nw", "se", "sw",
}

# Unit patterns: "9.8 m/s²", "100 kg", "273 K", "5 mol/L"
_UNIT_RE = re.compile(
    r'-?[\d.]+\s*(?:m|km|cm|mm|nm|µm|kg|g|mg|µg|lb|oz|'
    r's|ms|µs|ns|min|hr|h|'
    r'L|mL|mol|M|N|J|kJ|cal|kcal|eV|keV|MeV|W|kW|MW|'
    r'Pa|kPa|atm|bar|torr|mmHg|'
    r'K|°C|°F|'
    r'A|V|Ω|F|H|T|Wb|'
    r'm/s|km/h|m/s²|rad/s|Hz|kHz|MHz|GHz|'
    r'mol/L|g/mol|kg/m³|J/mol|'
    r'%|ppm|ppb)\b',
    re.IGNORECASE,
)

# Scientific notation: 3.0 × 10^8, 6.022e23
_SCI_NOTATION_RE = re.compile(
    r'-?[\d.]+\s*[×xX*]\s*10\s*\^|'
    r'-?[\d.]+[eE][+-]?\d+'
)

# Date patterns
_DATE_RE = re.compile(
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d|'
    r'\d{1,2}/\d{1,2}/\d{2,4}|'
    r'\d{4}-\d{2}-\d{2}|'
    r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|'
    r'\b(?:BC|AD|BCE|CE)\b|'
    r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    re.IGNORECASE,
)

# Ratio: 3:1, 2:3:5
_RATIO_RE = re.compile(r'^\d+\s*:\s*\d+(?:\s*:\s*\d+)*$')

# Bio sequences: DNA/RNA/protein
_BIO_SEQ_RE = re.compile(
    r"(?:5'|3')?-?[ATCGURYN]{4,}-?(?:3'|5')?|"
    r'\b(?:Ala|Arg|Asn|Asp|Cys|Glu|Gln|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)(?:-(?:Ala|Arg|Asn|Asp|Cys|Glu|Gln|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)){2,}',
    re.IGNORECASE,
)

# Coordinate/tuple: (3, 4), (x, y, z)
_COORD_RE = re.compile(r'^\(\s*-?[\d.]+\s*,\s*-?[\d.]+(?:\s*,\s*-?[\d.]+)*\s*\)$')

# Range/interval: [2, 7], (0, 1)
_RANGE_RE = re.compile(r'^[\[\(]\s*-?[\d.]+\s*,\s*-?[\d.]+\s*[\]\)]$')

# JSON
_JSON_RE = re.compile(r'^\s*\{.*\}\s*$', re.DOTALL)


def detect_traits(expected_answer: str, dataset_name: str = "") -> List[str]:
    """Detect all matching traits from the expected answer and dataset name.

    Returns a list of trait keys from the TRAITS dict. Multiple traits can
    fire for a single answer (composition, not classification).
    Zero LLM cost — purely regex + dataset lookup.
    """
    traits: Set[str] = set()

    # ── 1. Dataset hints (free) ──
    if dataset_name:
        ds = dataset_name.strip().lower()
        # Exact match
        if ds in _DATASET_TRAIT_HINTS:
            traits.update(_DATASET_TRAIT_HINTS[ds])
        else:
            # Partial/basename match
            for key, hint_traits in _DATASET_TRAIT_HINTS.items():
                if key in ds or ds in key:
                    traits.update(hint_traits)
                    break
            else:
                basename = ds.rsplit("/", 1)[-1]
                if basename in _DATASET_TRAIT_HINTS:
                    traits.update(_DATASET_TRAIT_HINTS[basename])

    # ── 2. Regex detection on expected answer ──
    if not expected_answer:
        traits.add("categorical_concept")
        return _dedupe_and_order(traits)

    ans = expected_answer.strip()
    ans_lower = ans.lower()

    # Boolean
    if ans_lower in ("true", "false", "yes", "no", "correct", "incorrect"):
        traits.add("boolean")

    # MCQ letter: A, B, C, D, (A), (B)
    if re.match(r'^\(?[A-Ea-e]\)?\.?$', ans):
        traits.add("mcq_letter")

    # Labeled choice: "Mutant 2", "Option B", etc.
    if re.match(_LABELED_WORDS, ans):
        traits.add("labeled_choice")

    # Roman numeral: II, III, IV
    if re.match(r'^[IVXivx]+\.?$', ans) and len(ans) <= 5:
        traits.add("roman_numeral")
        traits.add("labeled_choice")

    # Chemical formula
    if _CHEMICAL_RE.match(ans) and len(ans) >= 3:
        traits.add("chemical_formula")

    # Bio sequence
    if _BIO_SEQ_RE.search(ans):
        traits.add("bio_sequence")

    # Unit quantity
    if _UNIT_RE.search(ans):
        traits.add("unit_quantity")

    # Scientific notation
    if _SCI_NOTATION_RE.search(ans):
        traits.add("scientific_notation")

    # Date/time
    if _DATE_RE.search(ans):
        traits.add("date_time")

    # LaTeX expression
    if '\\' in ans:
        traits.add("expression_latex")

    # Algebraic expression (caret notation)
    if re.search(r'[a-zA-Z0-9]\^[a-zA-Z0-9{]', ans):
        traits.add("expression_algebraic")

    # Ratio
    if _RATIO_RE.match(ans):
        traits.add("ratio")

    # Coordinate/tuple
    if _COORD_RE.match(ans):
        traits.add("coordinate")
        traits.add("multi_part")

    # Range/interval
    if _RANGE_RE.match(ans):
        traits.add("range_interval")

    # Sequence/list (3+ comma-separated items or bracketed list)
    if re.match(r'^\[.*\]$', ans) and ',' in ans:
        traits.add("sequence_list")
    elif ans.count(',') >= 2:
        traits.add("sequence_list")

    # Ordering/ranking: contains > or "to" pattern for ranking
    if re.search(r'\s*>\s*', ans) and ans.count('>') >= 1:
        traits.add("ordering_ranking")

    # Multi-part: "x=2, y=3" or tuple-like
    if re.search(r'[a-z]\s*=\s*-?[\d.]+', ans, re.IGNORECASE) and ',' in ans:
        traits.add("multi_part")

    # JSON structured
    if _JSON_RE.match(ans):
        traits.add("json_structured")

    # Direction
    first_word = ans_lower.split()[0].rstrip('.,;:')
    if first_word in _DIRECTIONS or ans_lower in _DIRECTIONS:
        traits.add("categorical_direction")

    # Numeric checks (after more specific checks)
    clean = re.sub(r'[,$%]', '', ans)
    if re.match(r'^-?\d+$', clean):
        traits.add("numeric_integer")
        if ans.startswith('-'):
            traits.add("numeric_negative")
    elif re.match(r'^-?\d+\.\d+$', clean):
        traits.add("numeric_decimal")
        if ans.startswith('-'):
            traits.add("numeric_negative")
    elif re.match(r'^-?\d+\s*/\s*\d+$', clean):
        traits.add("numeric_fraction")
        if ans.startswith('-'):
            traits.add("numeric_negative")
    if '%' in ans and re.search(r'\d', ans):
        traits.add("numeric_percentage")

    # Named entity heuristic: capitalized multi-word (e.g. "Albert Einstein", "New York")
    # Exclude answers that are clearly structured data, not entity names
    words = ans.split()
    structural_traits = {"labeled_choice", "mcq_letter", "boolean", "coordinate",
                         "range_interval", "sequence_list", "ordering_ranking",
                         "multi_part", "json_structured", "numeric_integer",
                         "numeric_decimal", "numeric_fraction", "ratio"}
    if (len(words) >= 2 and
            all(w[0].isupper() for w in words if len(w) > 0 and w[0].isalpha()) and
            not (traits & structural_traits) and
            not re.match(r'^[\[\(\{]', ans)):
        traits.add("categorical_entity")

    # Explanation-bearing: long predicted answers often have explanations.
    # We always add this as a mild hint for the judge to extract core answers.
    traits.add("answer_with_explanation")

    # If nothing specific fired, add concept fallback
    if not traits - {"answer_with_explanation"}:
        traits.add("categorical_concept")

    return _dedupe_and_order(traits)


def _dedupe_and_order(traits: Set[str]) -> List[str]:
    """Order traits: critical/specific first, generic last."""
    # Priority tiers: higher = shown first in the prompt
    priority = {
        "labeled_choice": 0, "mcq_letter": 0, "roman_numeral": 0,
        "boolean": 1,
        "numeric_integer": 2, "numeric_decimal": 2, "numeric_fraction": 2,
        "numeric_percentage": 2, "numeric_negative": 2,
        "expression_latex": 3, "expression_algebraic": 3,
        "chemical_formula": 4, "bio_sequence": 4,
        "unit_quantity": 5, "scientific_notation": 5, "date_time": 5,
        "ratio": 5, "coordinate": 5, "range_interval": 5,
        "sequence_list": 6, "ordering_ranking": 6, "multi_part": 6,
        "json_structured": 7, "code_output": 7,
        "categorical_entity": 8, "categorical_direction": 8,
        "categorical_concept": 9,
        "answer_with_explanation": 10,  # always last — generic
    }
    return sorted(traits, key=lambda t: priority.get(t, 99))


# ──────────────────────────────────────────────────────────────────────────────
# LLM fallback — pattern-level cache + classify into EXISTING traits
# ──────────────────────────────────────────────────────────────────────────────
# Only fires when: (1) regex finds zero specific traits, AND
#                   (2) cfg["llm_trait_fallback"] is True.
# Output is bounded: LLM picks from the fixed set of 28 trait names.
# Cached at the pattern level so "Mutant 2" and "Mutant 17" share one entry.

# Thread-safe pattern cache: pattern_key → list of trait names
_llm_trait_cache: Dict[str, List[str]] = {}
_cache_lock = threading.Lock()

# Available trait names for the LLM to pick from (built once)
_TRAIT_MENU = "\n".join(
    f"  {key}: {trait.get('preamble', '')[:80]}"
    for key, trait in TRAITS.items()
    if key != "answer_with_explanation"
)


def _normalize_answer_pattern(answer: str) -> str:
    """Normalize an answer to a pattern key for caching.

    "Mutant 2", "Mutant 3", "Mutant 17" → "Word <N>"
    "42", "73", "1000" → "<integer>"
    "H2SO4", "NaCl" → "<chemical>"
    Anything else → first 50 chars lowercased with numbers replaced.
    """
    ans = answer.strip()
    if not ans:
        return "<empty>"

    # Replace all digit sequences with <N>
    pattern = re.sub(r'\d+', '<N>', ans)
    # Collapse whitespace
    pattern = re.sub(r'\s+', ' ', pattern).strip().lower()
    # Truncate for cache key sanity
    return pattern[:80]


def _build_trait_classify_prompt(expected_answer: str) -> str:
    """Build the prompt for LLM trait classification."""
    return f"""You are classifying an answer style for an evaluation judge.

Given this expected answer, pick 1-3 traits from the list below that best describe
how a judge should compare a predicted answer against this ground truth.

EXPECTED ANSWER: {expected_answer}

AVAILABLE TRAITS (pick by key name):
{_TRAIT_MENU}

Reply with ONLY the trait key names, comma-separated. Example: "numeric_integer, unit_quantity"
If none fit well, reply "categorical_concept".
"""


def _parse_trait_response(raw: str) -> List[str]:
    """Parse an LLM response into valid trait keys."""
    candidates = [t.strip().lower().replace(" ", "_") for t in raw.strip().split(",")]
    valid = [t for t in candidates if t in TRAITS]
    return valid if valid else ["categorical_concept"]


def _llm_classify_traits(expected_answer: str,
                         call_fn: Optional[callable] = None) -> List[str]:
    """Use a single cheap LLM call to classify the answer into existing traits.

    Picks from the fixed set of 28 traits — never invents new ones.
    Accepts a call_fn(prompt) → str callable. The caller provides this
    so the core pipeline doesn't depend on any specific LLM backend.
    Returns a list of trait keys, or empty list on failure.
    """
    if not call_fn:
        return []
    try:
        prompt = _build_trait_classify_prompt(expected_answer)
        raw = call_fn(prompt)
        return _parse_trait_response(raw)
    except Exception as e:
        logger.debug(f"LLM trait fallback failed: {e}")
        return []


def _get_llm_traits_cached(expected_answer: str,
                           call_fn: Optional[callable] = None) -> List[str]:
    """Get LLM-classified traits with pattern-level caching.

    Thread-safe. Same answer pattern only triggers one LLM call ever.
    """
    pattern_key = _normalize_answer_pattern(expected_answer)

    with _cache_lock:
        if pattern_key in _llm_trait_cache:
            return _llm_trait_cache[pattern_key]

    # Cache miss — make the LLM call (outside lock to avoid blocking)
    traits = _llm_classify_traits(expected_answer, call_fn)

    with _cache_lock:
        # Double-check (another thread may have filled it)
        if pattern_key not in _llm_trait_cache:
            _llm_trait_cache[pattern_key] = traits
            logger.debug(f"LLM trait cache: '{pattern_key}' → {traits}")
        return _llm_trait_cache[pattern_key]


# ──────────────────────────────────────────────────────────────────────────────
# Compose few-shot block from detected traits
# ──────────────────────────────────────────────────────────────────────────────

def _compose_blocks(traits: List[str]) -> str:
    """Build the few-shot text block from a list of trait keys."""
    blocks = []
    seen_examples: set = set()

    for trait_key in traits:
        trait = TRAITS.get(trait_key)
        if not trait:
            continue

        preamble = trait.get("preamble", "")
        examples = trait.get("examples", [])

        lines = []
        if preamble and preamble not in seen_examples:
            lines.append(preamble)
            seen_examples.add(preamble)

        for ex in examples:
            if ex not in seen_examples:
                lines.append(f"- {ex}")
                seen_examples.add(ex)

        if lines:
            blocks.append("\n".join(lines))

    if not blocks:
        return (
            "- GT: \"42\" vs PRED: \"42.0\" → YES (same value)\n"
            "- GT: \"Monday\" vs PRED: \"Tuesday\" → NO (different days)\n"
            "- GT: \"Option A\" vs PRED: \"Option B\" → NO (different options)"
        )

    return "\n\n".join(blocks)


def get_few_shot_examples(expected_answer: str, dataset_name: str = "",
                          cfg: Optional[Dict] = None,
                          call_fn: Optional[callable] = None) -> str:
    """Compose a few-shot examples block from all detected traits.

    Multiple traits fire → examples from all are combined.

    If cfg["llm_trait_fallback"] is True and regex finds no specific traits,
    a single LLM call classifies into existing traits (cached at pattern level).
    The caller provides call_fn(prompt) → str so this module stays backend-agnostic.

    Args:
        expected_answer: The ground truth answer.
        dataset_name: Optional HF dataset name for trait hints.
        cfg: Optional config dict. Key: "llm_trait_fallback" (bool, default False).
        call_fn: Optional callable(prompt: str) → str for LLM fallback.
                 The caller constructs this from whatever backend they use.

    Returns a string ready to inject into the judge prompt.
    """
    traits = detect_traits(expected_answer, dataset_name)

    # Check if regex found anything specific
    specific = [t for t in traits if t not in ("answer_with_explanation", "categorical_concept")]

    # LLM fallback: only if enabled AND regex found nothing specific AND callable provided
    if not specific and cfg and cfg.get("llm_trait_fallback") and call_fn:
        llm_traits = _get_llm_traits_cached(expected_answer, call_fn)
        if llm_traits:
            merged = set(traits) | set(llm_traits)
            traits = _dedupe_and_order(merged)

    return _compose_blocks(traits)


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible API (used by runner.py)
# ──────────────────────────────────────────────────────────────────────────────

def classify_answer_style(expected_answer: str, dataset_name: str = "") -> str:
    """Return the primary (highest-priority) trait as a category string.

    Kept for backward compatibility. Internally uses detect_traits().
    """
    traits = detect_traits(expected_answer, dataset_name)
    specific = [t for t in traits if t not in ("answer_with_explanation", "categorical_concept")]
    return specific[0] if specific else "categorical_concept"
