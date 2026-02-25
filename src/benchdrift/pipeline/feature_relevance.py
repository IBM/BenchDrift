"""
Taxonomy-driven two-level relevance ranking for BenchDrift transformations.

Level 1 — Axis ranking:
    Rank the 4-6 axes (linguistic, referential, pragmatic, structural,
    persona, long_context) by relevance to the problem.
    Uses a single LLM call when available, falls back to feature counting.

Level 2 — Within-axis selection:
    Distribute top-k slots proportionally across ranked axes (geometric decay).
    Within each axis, pick transformations by WITHIN_AXIS_AFFINITY feature match.

Two layers of feature detection:
  1. Regex features (~0 ms) — 20 boolean detectors + derived numerics
  2. LLM enrichment (optional, ~2-3 sec) — 20 deeper structural features
     detected via two parallel Ollama calls. Graceful fallback if unavailable.
"""

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Feature detectors  (regex, ~0 ms per problem)
# ---------------------------------------------------------------------------

FEATURE_DETECTORS: Dict[str, re.Pattern] = {
    'has_units': re.compile(
        r'\b(cm|mm|m|km|inch|inches|feet|ft|foot|mile|miles|mph|km/h|'
        r'hour|hours|minute|minutes|second|seconds|'
        r'dollar|dollars|cent|cents|pound|pounds|'
        r'kg|kilogram|kilograms|gram|grams|'
        r'liter|liters|litre|litres|gallon|gallons|ounce|ounces|'
        r'acre|acres|hectare|hectares|'
        r'°[CF]|fahrenheit|celsius)\b', re.IGNORECASE
    ),
    'has_temporal': re.compile(
        r'\b(AM|PM|a\.m\.|p\.m\.|o\'clock|'
        r'hour|hours|day|days|week|weeks|month|months|year|years|'
        r'today|yesterday|tomorrow|morning|evening|afternoon|night|'
        r'before|after|earlier|later|ago|since|until|'
        r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|'
        r'January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\b', re.IGNORECASE
    ),
    'has_named_entity': re.compile(
        r'\b[A-Z][a-z]{2,}\b'  # Capitalized words (names/places)
    ),
    'has_decimals': re.compile(r'\d+\.\d+'),
    'has_fractions': re.compile(r'\b\d+\s*/\s*\d+\b|½|⅓|⅔|¼|¾|⅕|⅖|⅗|⅘|⅙|⅚|⅛|⅜|⅝|⅞'),
    'has_money': re.compile(r'\$\s*\d|\d+\s*(dollar|cent|pound|euro|£|€)', re.IGNORECASE),
    'has_percentage': re.compile(r'\d+\s*%|percent', re.IGNORECASE),
    'has_negation': re.compile(
        r"\b(not|no|never|don't|doesn't|won't|can't|cannot|without|"
        r"none|neither|nor|hardly|barely|isn't|aren't|wasn't|weren't)\b",
        re.IGNORECASE
    ),
    'has_comparison': re.compile(
        r'\b(more|less|fewer|greater|larger|smaller|taller|shorter|'
        r'heavier|lighter|faster|slower|older|younger|'
        r'than|compared|ratio|twice|triple|half|double)\b', re.IGNORECASE
    ),
    'has_conditional': re.compile(
        r'\b(if|when|unless|provided|assuming|suppose|given that|'
        r'in case|whether)\b', re.IGNORECASE
    ),
    'has_geometry': re.compile(
        r'\b(triangle|rectangle|square|circle|sphere|cube|cylinder|cone|'
        r'pentagon|hexagon|polygon|parallelogram|trapezoid|rhombus|'
        r'angle|angles|degree|degrees|'
        r'area|perimeter|circumference|volume|surface area|'
        r'radius|diameter|hypotenuse|diagonal|'
        r'right angle|isosceles|equilateral|scalene)\b', re.IGNORECASE
    ),
    'has_probability': re.compile(
        r'\b(probability|probabilit|chance|odds|likely|likelihood|'
        r'dice|die|coin|flip|random|randomly|'
        r'event|outcome|outcomes|favorable|'
        r'deck|cards?|draw|drawn|lottery|'
        r'percent chance|expected value|expected number)\b', re.IGNORECASE
    ),
    'has_ratios': re.compile(
        r'\b(ratio|ratios|proportion|proportional|'
        r'rate|rates|per\b|for every|out of|'
        r'scale|scaled|factor|'
        r'to every|for each)\b', re.IGNORECASE
    ),
    'has_sequence': re.compile(
        r'\b(first|then|next|finally|afterwards|subsequently|'
        r'step \d|stage \d|'
        r'begins? by|starts? by|followed by|'
        r'after that|before that|in order)\b', re.IGNORECASE
    ),
    'has_equation': re.compile(
        r'\b(solve for|equation|equations|expression|expressions|'
        r'variable|variables|formula|'
        r'evaluate|simplify|factor|expand)\b|'
        r'[a-z]\s*[+\-*/]\s*[a-z]|'   # e.g. x + y
        r'[a-z]\s*=\s*\d',             # e.g. x = 5
        re.IGNORECASE
    ),
    'has_counting': re.compile(
        r'\b(how many|total|count|altogether|combined|'
        r'sum of|in all|number of|'
        r'each|every one|all together)\b', re.IGNORECASE
    ),
    'has_spatial': re.compile(
        r'\b(above|below|left|right|north|south|east|west|'
        r'between|beside|behind|front|back|'
        r'distance|position|direction|'
        r'adjacent|opposite|parallel|perpendicular|'
        r'top|bottom|inside|outside)\b', re.IGNORECASE
    ),
    'has_logical_connectives': re.compile(
        r'\b(therefore|thus|hence|consequently|'
        r'because|since|so that|in order to|'
        r'implies|implication|'
        r'either|or|neither|nor|'
        r'however|although|despite|nevertheless)\b', re.IGNORECASE
    ),
    'has_constraint': re.compile(
        r'\b(at least|at most|no more than|no less than|no fewer than|'
        r'maximum|minimum|exactly|precisely|'
        r'must be|cannot exceed|cannot be|'
        r'limit|bounded|within|range|'
        r'up to|not exceed|constraint)\b', re.IGNORECASE
    ),
    'has_rate_of_change': re.compile(
        r'\b(per hour|per day|per minute|per second|per week|per month|per year|'
        r'mph|km/h|m/s|ft/s|'
        r'speed|velocity|acceleration|'
        r'growth rate|rate of|'
        r'every hour|every day|every minute|every week|'
        r'each hour|each day|each minute|each week)\b', re.IGNORECASE
    ),
    'has_enumeration': re.compile(
        r'(?:'
        r'(?:^|\n)\s*[A-J][.)]\s|'          # A) or A. style options
        r'(?:^|\n)\s*\([A-Ja-j]\)\s|'       # (A) or (a) style
        r'(?:^|\n)\s*\d+[.)]\s|'            # 1) or 1. numbered items
        r'(?:^|\n)\s*\(\d+\)\s|'            # (1) style
        r'(?:^|\n)\s*[ivxIVX]+[.)]\s|'      # roman numeral items
        r'(?:^|\n)\s*[-*]\s|'               # bullet points
        r'(?:^|\n)\s*[a-j][.)]\s'           # lowercase letter options
        r')', re.MULTILINE
    ),
}


def get_problem_features(problem_text: str) -> Dict[str, object]:
    """Detect features in a problem using regex patterns and heuristics.

    Returns a dict of feature names to values (bool for pattern matches,
    int/float for derived numeric features).
    """
    features: Dict[str, object] = {}

    # Regex-based boolean features
    for name, pattern in FEATURE_DETECTORS.items():
        features[name] = bool(pattern.search(problem_text))

    # Derived numeric features
    numbers = re.findall(r'-?\b\d+\.?\d*\b', problem_text)
    num_numbers = len(numbers)
    features['num_numbers'] = num_numbers
    features['num_numbers_high'] = num_numbers > 3

    num_sentences = problem_text.count('.') + problem_text.count('?') + problem_text.count('!')
    features['num_sentences'] = num_sentences

    features['is_multi_step'] = num_numbers > 3 or num_sentences > 2
    features['is_word_problem'] = (
        num_sentences >= 2 and bool(re.search(r'[A-Z][a-z]', problem_text))
    )
    features['problem_length'] = len(problem_text)
    features['is_long'] = len(problem_text) > 300

    return features


# ---------------------------------------------------------------------------
# Taxonomy (replaces RELEVANCE_TABLE)
# ---------------------------------------------------------------------------

TAXONOMY: Dict[str, Dict] = {
    "linguistic": {
        "description": "Surface-level language reformulations that preserve meaning",
        "transformations": [
            "rephrasing", "narrative_style", "interrogative_expansion",
            "logical_formulation", "format_variation", "symbolic_representation",
            "programming_formulation", "grammar_correction",
            "politeness_variation", "passive_active_voice",
        ],
        "affinity_features": [
            "is_word_problem", "is_long", "has_equation", "has_logical_connectives",
        ],
    },
    "referential": {
        "description": "Changes to entities, units, domains, and numerical precision",
        "transformations": [
            "domain_shift", "unit_conversion", "precision_variation",
            "scale_extremes", "missing_context",
        ],
        "affinity_features": [
            "has_units", "has_money", "has_decimals", "has_fractions",
            "has_named_entity", "has_rate_of_change",
            "has_domain_specific", "has_implicit_conversion",
        ],
    },
    "pragmatic": {
        "description": "Framing, bias, and pragmatic meaning changes",
        "transformations": [
            "hypothetical_framing", "assumption_testing",
            "emotional_bias_injection", "anchoring_bias_test",
            "false_premise_detection", "ambiguous_phrasing",
            "verification_task", "uncertainty_markers",
        ],
        "affinity_features": [
            "has_conditional", "has_negation", "has_comparison",
            "has_constraint", "has_distractor", "has_abstract_reasoning",
            "has_verification", "has_estimation",
        ],
    },
    "structural": {
        "description": "Structural rearrangements, cognitive load, and meta-references",
        "transformations": [
            "cognitive_load_stress", "order_dependency",
            "step_by_step_decomposition", "irrelevant_context",
            "negation_complexity", "temporal_confusion",
            "meta_reference_loop", "implicit_constraint_test",
            "reverse_problem_formulation", "causal_framing",
            "enumeration_format_variation", "option_delimiter_variation",
            "option_order_shuffle",
        ],
        "affinity_features": [
            "is_multi_step", "has_sequence", "has_temporal",
            "has_backward_reasoning", "has_aggregation",
            "has_multi_entity_chain", "num_numbers_high",
            "has_inverse_problem", "has_causal_chain",
            "has_enumeration", "has_table_or_list",
        ],
    },
    "persona": {
        "description": "Domain persona re-framings",
        "transformations": [
            "scientist_persona", "artist_persona", "chef_persona",
            "detective_persona", "teacher_persona", "engineer_persona",
            "doctor_persona", "farmer_persona", "athlete_persona",
            "musician_persona",
        ],
        "affinity_features": [
            "is_word_problem", "has_named_entity", "has_units",
            "has_geometry", "has_probability",
        ],
    },
    "long_context": {
        "description": "Structural variations for long prompts (>300 chars)",
        "transformations": [
            "long_context.format.quotes", "long_context.format.whitespace",
            "long_context.format.case", "long_context.positioning.sections",
            "long_context.positioning.paragraphs", "long_context.content.removal",
            "long_context.quality.clarity", "long_context.quality.completeness",
            "long_context.quality.ambiguity", "long_context.style.redundancy",
            "long_context.style.formality", "long_context.style.complexity",
        ],
        "affinity_features": ["is_long"],
        "gate": "is_long",  # Only include when problem is_long
    },
    "constraint_targeted": {
        "description": "Constraint-aware variations that identify and reformulate specific given values, conditions, and relationships in the problem",
        "transformations": [
            "constraint_format_variation",
            "constraint_reordering",
            "constraint_explicitness",
            "constraint_compound_variation",
            "constraint_redundancy",
        ],
        "affinity_features": [
            "has_constraint", "has_conditional", "has_comparison",
            "num_numbers_high", "is_multi_step", "has_units",
            "has_money", "has_rate_of_change",
        ],
    },
}

# ---------------------------------------------------------------------------
# Derived constants from TAXONOMY
# ---------------------------------------------------------------------------

ALL_AXIS_NAMES: FrozenSet[str] = frozenset(TAXONOMY.keys())

TRANSFORMATION_TO_AXIS: Dict[str, str] = {}
for _ax_name, _ax_def in TAXONOMY.items():
    for _t in _ax_def["transformations"]:
        TRANSFORMATION_TO_AXIS[_t] = _ax_name


def parse_axes(axes_str: str) -> Set[str]:
    """Parse a comma-separated axis specification string into a set of axis names.

    Supports:
        "linguistic,structural"           -> {linguistic, structural}
        "all"                             -> all 7 axes
        "all,-persona"                    -> all except persona
        "all,-persona,-long_context"      -> all except persona and long_context

    Raises ValueError for unknown axis names.
    """
    parts = [p.strip() for p in axes_str.split(',') if p.strip()]
    result: Set[str] = set()
    for p in parts:
        if p == 'all':
            result = set(ALL_AXIS_NAMES)
        elif p.startswith('-'):
            name = p[1:]
            if name not in ALL_AXIS_NAMES:
                raise ValueError(
                    f"Unknown axis: '{name}'. Valid: {', '.join(sorted(ALL_AXIS_NAMES))}")
            result.discard(name)
        elif p in ALL_AXIS_NAMES:
            result.add(p)
        else:
            raise ValueError(
                f"Unknown axis: '{p}'. Valid: {', '.join(sorted(ALL_AXIS_NAMES))}")
    return result


# ---------------------------------------------------------------------------
# Within-axis affinity (for within-axis selection without LLM)
# ---------------------------------------------------------------------------
# Only transformations where features strongly differentiate them from siblings.
# Transformations not listed get a neutral score within their axis.

WITHIN_AXIS_AFFINITY: Dict[str, List[str]] = {
    # Referential (most feature-sensitive)
    "unit_conversion":        ["has_units", "has_money", "has_rate_of_change", "has_implicit_conversion"],
    "precision_variation":    ["has_decimals", "has_fractions", "has_percentage", "has_estimation"],
    "domain_shift":           ["has_named_entity", "is_word_problem", "has_domain_specific", "has_analogy"],
    "scale_extremes":         ["has_units", "num_numbers_high", "has_ratios"],
    "missing_context":        ["has_constraint", "has_backward_reasoning"],
    # Structural
    "temporal_confusion":     ["has_temporal", "has_rate_of_change", "has_causal_chain"],
    "order_dependency":       ["is_multi_step", "has_sequence", "has_symmetry"],
    "negation_complexity":    ["has_negation", "has_logical_connectives", "has_conditional_branches"],
    "cognitive_load_stress":  ["is_multi_step", "num_numbers_high", "has_multi_entity_chain", "has_conditional_branches"],
    "implicit_constraint_test": ["has_constraint", "has_backward_reasoning", "has_conditional", "has_inverse_problem"],
    "step_by_step_decomposition": ["is_multi_step", "has_aggregation", "has_nested_operations"],
    "irrelevant_context":     ["is_multi_step", "has_distractor"],
    "meta_reference_loop":    ["has_abstract_reasoning", "has_counting", "has_equation"],
    "reverse_problem_formulation": ["has_inverse_problem", "has_backward_reasoning", "has_optimization"],
    "causal_framing":         ["has_causal_chain", "has_temporal", "has_logical_connectives"],
    "enumeration_format_variation": ["has_enumeration", "has_table_or_list"],
    "option_delimiter_variation":   ["has_enumeration", "has_table_or_list"],
    "option_order_shuffle":         ["has_enumeration", "has_table_or_list", "has_symmetry"],
    # Linguistic
    "symbolic_representation": ["has_equation", "has_decimals", "num_numbers_high", "has_nested_operations"],
    "programming_formulation": ["has_equation", "is_multi_step"],
    "format_variation":       ["is_long", "has_table_or_list", "has_symmetry"],
    "rephrasing":             ["is_word_problem", "is_long", "has_definitional"],
    "logical_formulation":    ["has_logical_connectives", "num_numbers_high", "is_multi_step", "has_set_operations"],
    "interrogative_expansion": ["is_multi_step", "has_counting"],
    "politeness_variation":   ["is_word_problem", "has_named_entity"],
    "passive_active_voice":   ["is_word_problem", "has_causal_chain", "has_sequence"],
    # Pragmatic
    "anchoring_bias_test":    ["has_money", "num_numbers_high", "has_distractor"],
    "false_premise_detection": ["has_negation", "has_logical_connectives", "has_verification"],
    "assumption_testing":     ["has_conditional", "has_constraint", "has_multiple_correct"],
    "hypothetical_framing":   ["has_conditional", "has_abstract_reasoning"],
    "emotional_bias_injection": ["is_word_problem", "has_named_entity"],
    "ambiguous_phrasing":     ["has_comparison", "has_spatial", "has_multiple_correct"],
    "verification_task":      ["has_verification", "has_definitional", "has_equation"],
    "uncertainty_markers":    ["has_estimation", "has_decimals", "has_probability"],
    # Persona
    "artist_persona":         ["has_geometry", "has_spatial"],
    "scientist_persona":      ["has_units", "has_decimals", "has_probability"],
    "engineer_persona":       ["has_units", "has_geometry", "has_equation"],
    "athlete_persona":        ["has_rate_of_change", "has_spatial"],
    "doctor_persona":         ["has_percentage", "has_ratios"],
    "musician_persona":       ["has_ratios", "has_sequence"],
    "chef_persona":           ["has_units", "has_ratios", "has_fractions"],
    "detective_persona":      ["has_named_entity", "has_backward_reasoning"],
    # Constraint-targeted
    "constraint_format_variation":     ["has_units", "has_money", "has_decimals", "has_fractions", "num_numbers_high"],
    "constraint_reordering":           ["is_multi_step", "has_sequence", "has_temporal", "has_causal_chain"],
    "constraint_explicitness":         ["has_constraint", "has_conditional", "has_backward_reasoning", "has_implicit_conversion"],
    "constraint_compound_variation":   ["num_numbers_high", "is_multi_step", "has_units", "has_conditional_branches"],
    "constraint_redundancy":           ["has_constraint", "has_comparison", "has_negation", "has_verification"],
}


# ---------------------------------------------------------------------------
# Unmapped LLM feature → transformation mapping
# ---------------------------------------------------------------------------

LLM_FEATURE_TO_TRANSFORMATIONS: Dict[str, List[str]] = {
    # Original 8
    "has_multi_entity_chain": ["cognitive_load_stress"],
    "has_implicit_conversion": ["unit_conversion", "assumption_testing"],
    "has_aggregation": ["step_by_step_decomposition"],
    "has_distractor": ["irrelevant_context", "anchoring_bias_test"],
    "has_domain_specific": ["domain_shift"],
    "has_abstract_reasoning": ["logical_formulation", "meta_reference_loop"],
    "has_backward_reasoning": ["implicit_constraint_test"],
    "has_table_or_list": ["format_variation", "order_dependency", "enumeration_format_variation", "option_delimiter_variation", "option_order_shuffle"],
    # New 12
    "has_causal_chain": ["causal_framing", "temporal_confusion"],
    "has_analogy": ["domain_shift", "rephrasing"],
    "has_verification": ["verification_task", "false_premise_detection"],
    "has_inverse_problem": ["reverse_problem_formulation", "implicit_constraint_test"],
    "has_multiple_correct": ["ambiguous_phrasing", "assumption_testing"],
    "has_conditional_branches": ["negation_complexity", "cognitive_load_stress", "constraint_explicitness"],
    "has_nested_operations": ["step_by_step_decomposition", "symbolic_representation", "constraint_compound_variation"],
    "has_estimation": ["precision_variation", "uncertainty_markers"],
    "has_definitional": ["rephrasing", "domain_shift"],
    "has_set_operations": ["logical_formulation", "symbolic_representation"],
    "has_optimization": ["reverse_problem_formulation", "cognitive_load_stress"],
    "has_symmetry": ["order_dependency", "format_variation"],
}


# ---------------------------------------------------------------------------
# Level 1 — Axis ranking
# ---------------------------------------------------------------------------

_AXIS_RANKING_SYSTEM_PROMPT = (
    "You rank problem axes. Return ONLY a JSON array of axis names, "
    "most relevant first. No explanation."
)

_AXIS_RANKING_USER_TEMPLATE = """\
Given this math/reasoning problem and its detected features, rank these axes \
from MOST to LEAST relevant for generating meaning-preserving variations that \
could cause model errors.

AXES:
{axes_block}

PROBLEM: {problem}
ACTIVE FEATURES: {features_summary}

Return ONLY a JSON array like ["structural","referential","pragmatic","linguistic"]. \
No explanation, no markdown fences."""


def _summarize_features(features: Dict[str, object]) -> str:
    """Compact string of active features for LLM prompt context."""
    active = [k for k, v in features.items()
              if v and k not in ('num_numbers', 'num_sentences', 'problem_length')]
    return ", ".join(active) if active else "none"


def _parse_axis_ranking(raw: str, valid_axes: List[str]) -> Optional[List[str]]:
    """Parse JSON array, comma-separated, or numbered list of axis names."""
    if not raw:
        return None

    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = [a.strip().lower() for a in parsed if isinstance(a, str)]
            result = [a for a in result if a in valid_axes]
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try comma-separated
    parts = re.split(r'[,\n]+', text)
    result = []
    for part in parts:
        # Strip numbering like "1. " or "1) "
        cleaned = re.sub(r'^\d+[.)]\s*', '', part.strip()).strip().strip('"\'').lower()
        if cleaned in valid_axes:
            result.append(cleaned)
    return result if result else None


def _rank_axes_by_features(features: Dict[str, object],
                           valid_axes: List[str]) -> List[str]:
    """Fallback: count matching affinity_features per axis, sort descending."""
    scores = {}
    for axis_name in valid_axes:
        axis = TAXONOMY[axis_name]
        aff_features = axis.get("affinity_features", [])
        count = sum(1 for f in aff_features if features.get(f))
        # Normalize by number of affinity features to avoid bias toward axes with more features
        total = len(aff_features) if aff_features else 1
        scores[axis_name] = count / total
    # Sort by score descending, then alphabetically for stability
    ranked = sorted(valid_axes, key=lambda a: (-scores[a], a))
    return ranked


def rank_axes_with_llm(
    problem_text: str,
    features: Dict[str, object],
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen3:8b",
    timeout: float = 8.0,
    include_personas: bool = True,
    enabled_axes: Optional[Set[str]] = None,
    call_fn=None,
) -> List[str]:
    """One LLM call: rank axes by relevance. Returns ['structural', 'referential', ...].

    Args:
        call_fn: Optional callable(system_prompt, user_prompt) -> str.
            If provided, uses this instead of direct requests.post to Ollama.
            This allows the caller to inject any LLM backend.

    Falls back to _rank_axes_by_features() on failure.
    """
    valid_axes = _get_valid_axes(features, include_personas, enabled_axes=enabled_axes)

    # Build axes block for prompt
    axes_lines = []
    for ax in valid_axes:
        desc = TAXONOMY[ax]["description"]
        axes_lines.append(f"- {ax}: {desc}")
    axes_block = "\n".join(axes_lines)

    features_summary = _summarize_features(features)
    user_prompt = _AXIS_RANKING_USER_TEMPLATE.format(
        axes_block=axes_block,
        problem=problem_text[:500],  # Truncate long problems
        features_summary=features_summary,
    )

    # --- call_fn path: backend-agnostic ---
    if call_fn:
        try:
            content = call_fn(_AXIS_RANKING_SYSTEM_PROMPT, user_prompt)
            parsed = _parse_axis_ranking(content, valid_axes)
            if parsed and len(parsed) >= 2:
                for ax in valid_axes:
                    if ax not in parsed:
                        parsed.append(ax)
                return parsed
        except Exception:
            pass
        return _rank_axes_by_features(features, valid_axes)

    # --- Fallback: direct requests.post to Ollama (batch pipeline compat) ---
    try:
        import requests
    except ImportError:
        return _rank_axes_by_features(features, valid_axes)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _AXIS_RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
        },
        "think": False,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{ollama_base_url}/api/chat",
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        parsed = _parse_axis_ranking(content, valid_axes)
        if parsed and len(parsed) >= 2:
            for ax in valid_axes:
                if ax not in parsed:
                    parsed.append(ax)
            return parsed
    except Exception:
        pass

    return _rank_axes_by_features(features, valid_axes)


def _get_valid_axes(features: Dict[str, object],
                    include_personas: bool = True,
                    enabled_axes: Optional[Set[str]] = None) -> List[str]:
    """Return axes valid for this problem (respecting gates, persona flag, and enabled set).

    Args:
        features: Problem feature dict.
        include_personas: Legacy flag — ignored when enabled_axes is provided.
        enabled_axes: If provided, only axes in this set are considered.
            Gate checks (e.g. long_context requires is_long) still apply on top.
    """
    valid = []
    for axis_name, axis_def in TAXONOMY.items():
        # If enabled_axes is provided, skip axes not in the set
        if enabled_axes is not None:
            if axis_name not in enabled_axes:
                continue
        else:
            # Legacy path: use include_personas flag
            if axis_name == "persona" and not include_personas:
                continue
        # Check gate (e.g. long_context only if is_long)
        gate = axis_def.get("gate")
        if gate and not features.get(gate):
            continue
        valid.append(axis_name)
    return valid


# ---------------------------------------------------------------------------
# Level 2 — Slot distribution + within-axis selection
# ---------------------------------------------------------------------------

_GEOMETRIC_WEIGHTS = [1.0, 0.6, 0.36, 0.22, 0.13, 0.08]


def distribute_slots(ranked_axes: List[str], top_k: int) -> Dict[str, int]:
    """Geometric decay: every axis gets at least 1 slot. Remainder to top axis."""
    n_axes = len(ranked_axes)
    if n_axes == 0:
        return {}
    if top_k <= n_axes:
        # Each axis gets 1 slot, truncate from the bottom
        return {ax: 1 for ax in ranked_axes[:top_k]}

    # Assign weights (extend with 0.05 if more than 6 axes)
    weights = []
    for i in range(n_axes):
        w = _GEOMETRIC_WEIGHTS[i] if i < len(_GEOMETRIC_WEIGHTS) else 0.05
        weights.append(w)

    total_weight = sum(weights)
    distributable = top_k - n_axes  # remaining after giving each axis 1

    slots = {}
    assigned = 0
    for i, ax in enumerate(ranked_axes):
        extra = round(distributable * weights[i] / total_weight)
        slots[ax] = 1 + extra
        assigned += 1 + extra

    # Fix rounding errors — add/remove from top axis
    diff = top_k - sum(slots.values())
    if diff != 0:
        slots[ranked_axes[0]] = max(1, slots[ranked_axes[0]] + diff)

    return slots


def select_within_axis(axis_name: str, n_slots: int,
                       features: Dict[str, object],
                       available_transformations: Optional[List[str]] = None) -> List[str]:
    """Pick n_slots transformations from axis.

    Score by counting matching WITHIN_AXIS_AFFINITY features.
    Ties broken by canonical order in TAXONOMY.
    If available_transformations is given, only consider those.
    """
    axis_def = TAXONOMY.get(axis_name)
    if not axis_def:
        return []

    canonical = axis_def["transformations"]
    if available_transformations is not None:
        # Only consider transformations that exist in the engine's available set
        canonical = [t for t in canonical if t in available_transformations]

    if not canonical:
        return []

    # Score each transformation
    scored = []
    for i, trans in enumerate(canonical):
        affinity_feats = WITHIN_AXIS_AFFINITY.get(trans, [])
        if affinity_feats:
            match_count = sum(1 for f in affinity_feats if features.get(f))
            score = match_count / len(affinity_feats)
        else:
            score = 0.5  # neutral for unlisted transformations
        scored.append((trans, score, i))  # i for canonical order tiebreak

    # Sort by score desc, then canonical order asc
    scored.sort(key=lambda x: (-x[1], x[2]))

    return [trans for trans, _score, _i in scored[:n_slots]]


# ---------------------------------------------------------------------------
# Top-level two-level ranking API
# ---------------------------------------------------------------------------

def rank_transformations_two_level(
    problem_text: str,
    features: Dict[str, object],
    transformation_types: Dict[str, dict],
    top_k: int = 10,
    include_personas: bool = True,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen3:8b",
    pre_ranked_axes: Optional[List[str]] = None,
    enabled_axes: Optional[Set[str]] = None,
) -> List[Tuple[str, float, str]]:
    """Main two-level API. Returns [(trans_name, score, axis_name), ...].

    If pre_ranked_axes provided, skips LLM call (reuses cached result).
    If enabled_axes provided, only those axes are considered (gate checks still apply).
    """
    available_set = set(transformation_types.keys())

    # Level 1: Axis ranking
    if pre_ranked_axes:
        ranked_axes = pre_ranked_axes
    else:
        ranked_axes = rank_axes_with_llm(
            problem_text, features, ollama_base_url, model,
            include_personas=include_personas,
            enabled_axes=enabled_axes,
        )

    # Filter to axes that have at least one available transformation
    valid_ranked = []
    for ax in ranked_axes:
        axis_trans = TAXONOMY.get(ax, {}).get("transformations", [])
        if any(t in available_set for t in axis_trans):
            valid_ranked.append(ax)

    if not valid_ranked:
        # Fallback: return all available transformations with neutral scores
        return [(t, 0.5, "unknown") for t in transformation_types]

    # Level 2: Slot distribution
    slots = distribute_slots(valid_ranked, top_k)

    # Within-axis selection
    result = []
    available_list = list(available_set)

    for axis_name in valid_ranked:
        n = slots.get(axis_name, 0)
        if n <= 0:
            continue
        selected = select_within_axis(axis_name, n, features, available_list)
        # Score: axis rank position determines base score
        axis_rank = valid_ranked.index(axis_name)
        base_score = max(0.1, 1.0 - axis_rank * 0.15)
        for trans in selected:
            # Add within-axis affinity bonus
            affinity_feats = WITHIN_AXIS_AFFINITY.get(trans, [])
            if affinity_feats:
                bonus = sum(1 for f in affinity_feats if features.get(f)) / len(affinity_feats) * 0.2
            else:
                bonus = 0.0
            score = min(1.0, base_score + bonus)
            result.append((trans, round(score, 3), axis_name))
            # Remove from available to avoid duplicates across axes
            if trans in available_list:
                available_list.remove(trans)

    return result


def find_unmapped_features(active_llm_features: Dict[str, bool],
                           selected_transformations: List[str]) -> List[str]:
    """Return LLM features that are active but whose mapped transformations weren't selected."""
    selected_set = set(selected_transformations)
    unmapped = []
    for feat, active in active_llm_features.items():
        if not active:
            continue
        mapped_trans = LLM_FEATURE_TO_TRANSFORMATIONS.get(feat, [])
        if mapped_trans and not any(t in selected_set for t in mapped_trans):
            unmapped.append(feat)
    return unmapped


# ---------------------------------------------------------------------------
# Backward-compatible API (used by batch pipeline)
# ---------------------------------------------------------------------------

def rank_transformations(
    problem_text: str,
    transformation_types: Dict[str, dict],
    enabled_axes: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """Rank all transformations by relevance to a problem. ~0 ms.

    Uses taxonomy-driven two-level ranking with feature-based axis ordering
    (no LLM call). Backward-compatible signature for the batch pipeline.

    Args:
        problem_text: The problem string to analyze.
        transformation_types: Dict of {name: config} from the engine.
        enabled_axes: If provided, only these axes are considered.

    Returns:
        List of (name, score) tuples sorted by decreasing relevance.
    """
    features = get_problem_features(problem_text)
    include_personas = any(k.endswith("_persona") for k in transformation_types)
    valid_axes = _get_valid_axes(features, include_personas, enabled_axes=enabled_axes)
    ranked_axes = _rank_axes_by_features(features, valid_axes)

    # Use full two-level ranking with feature-based axes (no LLM)
    # Ask for all transformations (top_k = len) to produce a full ranking
    top_k = len(transformation_types)
    result_3 = rank_transformations_two_level(
        problem_text, features, transformation_types,
        top_k=top_k, include_personas=include_personas,
        pre_ranked_axes=ranked_axes,
        enabled_axes=enabled_axes,
    )

    # Convert to 2-tuple format and add any transformations not covered by taxonomy
    seen = set()
    result = []
    for trans, score, _axis in result_3:
        if trans not in seen:
            result.append((trans, score))
            seen.add(trans)

    # Append any transformations in transformation_types not covered by TAXONOMY
    for name in transformation_types:
        if name not in seen:
            result.append((name, 0.3))
            seen.add(name)

    # Sort by score descending, name ascending for stability
    result.sort(key=lambda x: (-x[1], x[0]))
    return result


def select_top_k(
    problem_text: str,
    transformation_types: Dict[str, dict],
    k: int,
) -> List[str]:
    """Return the top-k most relevant transformation names. ~0 ms.

    Args:
        k: Number of transformations to select. 0 means all.
    """
    ranked = rank_transformations(problem_text, transformation_types)
    if k > 0:
        ranked = ranked[:k]
    return [name for name, _score in ranked]


def rank_transformations_with_features(
    features: Dict[str, object],
    transformation_types: Dict[str, dict],
    enabled_axes: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """Rank transformations using pre-computed features (regex + optionally LLM).

    Like rank_transformations() but takes a features dict directly instead of
    computing it from problem text. Use this when you've already called
    get_problem_features() or get_enriched_features().
    """
    include_personas = any(k.endswith("_persona") for k in transformation_types)
    valid_axes = _get_valid_axes(features, include_personas, enabled_axes=enabled_axes)
    ranked_axes = _rank_axes_by_features(features, valid_axes)

    top_k = len(transformation_types)
    result_3 = rank_transformations_two_level(
        "",  # no problem text needed when features are pre-computed
        features, transformation_types,
        top_k=top_k, include_personas=include_personas,
        pre_ranked_axes=ranked_axes,
        enabled_axes=enabled_axes,
    )

    seen = set()
    result = []
    for trans, score, _axis in result_3:
        if trans not in seen:
            result.append((trans, score))
            seen.add(trans)

    for name in transformation_types:
        if name not in seen:
            result.append((name, 0.3))
            seen.add(name)

    result.sort(key=lambda x: (-x[1], x[0]))
    return result


# ---------------------------------------------------------------------------
# LLM enrichment layer  (optional, ~2-3 sec via Ollama)
# ---------------------------------------------------------------------------

# The 20 features that require LLM understanding (detected in two parallel batches)
LLM_FEATURE_NAMES = [
    # Batch 1 (original 8)
    "has_multi_entity_chain",
    "has_implicit_conversion",
    "has_aggregation",
    "has_distractor",
    "has_domain_specific",
    "has_abstract_reasoning",
    "has_backward_reasoning",
    "has_table_or_list",
    # Batch 2 (12 new)
    "has_causal_chain",
    "has_analogy",
    "has_verification",
    "has_inverse_problem",
    "has_multiple_correct",
    "has_conditional_branches",
    "has_nested_operations",
    "has_estimation",
    "has_definitional",
    "has_set_operations",
    "has_optimization",
    "has_symmetry",
]

LLM_FEATURE_BATCH_1 = LLM_FEATURE_NAMES[:10]
LLM_FEATURE_BATCH_2 = LLM_FEATURE_NAMES[10:]

_LLM_SYSTEM_PROMPT = (
    "You are a strict math problem feature detector. Return ONLY a JSON object "
    "with boolean values. Be STRICT — only mark true when the definition clearly applies."
)

_LLM_USER_TEMPLATE_BATCH1 = """\
Analyze this problem for EXACTLY these 10 features. Be STRICT — read each \
definition carefully.

FEATURES:

1. has_multi_entity_chain: THREE or more entities where each depends on a previous one.
   TRUE example: "A gives 5 to B, B gives half to C, C gives 2 to D" (A→B→C→D chain)
   FALSE example: "A has 5, B has 3. How many total?" (independent, no chain)

2. has_implicit_conversion: Solving REQUIRES a conversion NOT stated in the problem.
   TRUE example: "She works 2 hours and earns $5/minute" (must convert hours→minutes)
   FALSE example: "She buys 3 items at $5 each" (no conversion needed)

3. has_aggregation: Answer requires combining results from TWO+ independent sub-calculations.
   TRUE example: "Find total cost: 3 shirts at $20 and 2 pants at $30" (two separate calcs merged)
   FALSE example: "What is 3 * 20?" (single calculation)

4. has_distractor: Contains a number or fact that is NOT needed for the answer.
   TRUE example: "A 50-foot bridge holds 5000 lbs. Cars weigh 2500 lbs. How many cars?" (50-foot is irrelevant)
   FALSE example: "A bridge holds 5000 lbs. Cars weigh 2500 lbs. How many cars?" (all info used)

5. has_domain_specific: Requires specialized knowledge BEYOND basic arithmetic.
   TRUE example: "What is the pH of a 0.01M HCl solution?" (chemistry knowledge)
   FALSE example: "A store sells apples for $2 each" (basic math only)

6. has_abstract_reasoning: Requires reasoning about patterns, logic, or relationships, NOT just computation.
   TRUE example: "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles?" (logic)
   FALSE example: "If x = 5 and y = 3, what is x + y?" (direct computation)

7. has_backward_reasoning: Must work BACKWARDS from a result to find an input.
   TRUE example: "After giving away 1/3 of her stickers, Mary has 20. How many did she start with?" (backward from 20)
   FALSE example: "Mary has 30 stickers and gives away 1/3. How many remain?" (forward calculation)

8. has_table_or_list: Data is presented in tabular form, a bullet list, or an enumerated list.
   TRUE example: "Item | Price\\nApple | $2\\nBanana | $1" (table format)
   FALSE example: "Apples cost $2 and bananas cost $1" (prose, not tabular)

9. has_causal_chain: Problem states explicit cause→effect relationship requiring causal reasoning.
   TRUE example: "Because it rained, the tank filled 5cm. How much after 3 days?" (cause→effect)
   FALSE example: "A tank fills 5cm per day. How much after 3 days?" (no causal link stated)

10. has_analogy: Problem uses analogy or proportional comparison ("A is to B as C is to D").
    TRUE example: "If 3 workers build 1 wall, how many walls can 9 workers build?" (proportional comparison)
    FALSE example: "3 workers build walls. How many walls do 9 workers build in 2 days?" (direct calculation)

Problem: {problem}

Return ONLY a JSON object like {{"has_multi_entity_chain": false, ...}} with all 10 keys."""

_LLM_USER_TEMPLATE_BATCH2 = """\
Analyze this problem for EXACTLY these 10 features. Be STRICT — read each \
definition carefully.

FEATURES:

1. has_verification: Problem asks to verify, check, or validate a given claim or answer.
   TRUE example: "Is it true that 15+25=40?" (asks to verify)
   FALSE example: "What is 15+25?" (asks to compute, not verify)

2. has_inverse_problem: Must find an input given an output (not just backward reasoning).
   TRUE example: "A number doubled then +3 gives 17. Find the number." (find input from output)
   FALSE example: "Double 7 then add 3." (forward computation)

3. has_multiple_correct: Problem might have multiple valid answers or solution approaches.
   TRUE example: "Find two numbers that sum to 10" (many valid pairs)
   FALSE example: "What is 5+5?" (single answer)

4. has_conditional_branches: Problem has if/else branching where different conditions lead to different calculations.
   TRUE example: "If x>5, add 10; otherwise subtract 3. x=7." (branching logic)
   FALSE example: "Add 10 to x=7." (no branching)

5. has_nested_operations: THREE+ levels of nested math operations (not just multi-step).
   TRUE example: "sqrt((3^2 + 4^2))" (3 levels: power, addition, sqrt)
   FALSE example: "3^2 + 4^2" (only 2 levels)

6. has_estimation: Problem involves approximation, rounding, or "about how much".
   TRUE example: "Approximately how many seconds in a year?" (asks for estimate)
   FALSE example: "How many seconds in 60 minutes?" (exact calculation)

7. has_definitional: Problem requires knowing a definition (what IS a prime number, a median, etc.).
   TRUE example: "What is the median of 3,7,1,9,5?" (must know what median means)
   FALSE example: "What is the middle value of 1,3,5,7,9?" (definition already given)

8. has_set_operations: Problem involves sets, groups, or Venn diagram logic.
   TRUE example: "20 play soccer, 15 play tennis, 8 play both. How many play at least one?" (set union)
   FALSE example: "20+15=?" (plain addition)

9. has_optimization: Problem asks for maximum, minimum, or best allocation.
   TRUE example: "Maximize profit given 100 units and costs..." (optimization)
   FALSE example: "What is the total cost?" (no optimization)

10. has_symmetry: Problem has symmetrical structure that could be exploited or broken.
    TRUE example: "A palindrome number has digits that sum to 10." (symmetry in structure)
    FALSE example: "A number has digits summing to 10." (no symmetry aspect)

Problem: {problem}

Return ONLY a JSON object like {{"has_verification": false, ...}} with all 10 keys."""

# Keep backward-compatible alias
_LLM_USER_TEMPLATE = _LLM_USER_TEMPLATE_BATCH1


def _parse_llm_features(raw: str) -> Dict[str, bool]:
    """Parse LLM response into feature dict. Handles markdown fences, partial JSON."""
    if not raw:
        return {}

    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {}

    # Filter to known feature names, coerce to bool
    result = {}
    for key in LLM_FEATURE_NAMES:
        if key in parsed:
            result[key] = bool(parsed[key])
    return result


def _call_ollama_batch(
    problem_text: str,
    template: str,
    ollama_base_url: str,
    model: str,
    timeout: float,
    call_fn=None,
) -> Dict[str, bool]:
    """Run one LLM feature detection batch. Internal helper.

    Args:
        call_fn: Optional callable(system_prompt, user_prompt) -> str.
            If provided, uses this instead of direct requests.post.
    """
    user_prompt = template.format(problem=problem_text)

    # --- call_fn path: backend-agnostic ---
    if call_fn:
        try:
            content = call_fn(_LLM_SYSTEM_PROMPT, user_prompt)
            return _parse_llm_features(content)
        except Exception:
            return {}

    # --- Fallback: direct requests.post to Ollama ---
    try:
        import requests
    except ImportError:
        return {}

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 384,
        },
        "think": False,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{ollama_base_url}/api/chat",
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _parse_llm_features(content)
    except Exception:
        return {}


def enrich_features_with_llm(
    problem_text: str,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen3:8b",
    timeout: float = 10.0,
    call_fn=None,
) -> Dict[str, bool]:
    """Detect 20 deeper structural features via two parallel LLM calls.

    Args:
        call_fn: Optional callable(system_prompt, user_prompt) -> str.
            If provided, uses this instead of direct requests.post to Ollama.
            This allows the caller to inject any LLM backend.

    Returns dict of {feature_name: bool}. Returns {} on any failure
    (network error, timeout, bad JSON, Ollama not running).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    merged: Dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_b1 = executor.submit(
            _call_ollama_batch, problem_text,
            _LLM_USER_TEMPLATE_BATCH1, ollama_base_url, model, timeout,
            call_fn,
        )
        future_b2 = executor.submit(
            _call_ollama_batch, problem_text,
            _LLM_USER_TEMPLATE_BATCH2, ollama_base_url, model, timeout,
            call_fn,
        )

        for future in as_completed([future_b1, future_b2]):
            try:
                result = future.result()
                merged.update(result)
            except Exception:
                pass

    return merged


def get_enriched_features(
    problem_text: str,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen3:8b",
    timeout: float = 10.0,
) -> Dict[str, object]:
    """Merge regex features + LLM features into one dict.

    Regex features are always present. LLM features are added if the call
    succeeds, otherwise the dict just contains regex features.
    """
    features = get_problem_features(problem_text)
    llm_features = enrich_features_with_llm(
        problem_text, ollama_base_url, model, timeout
    )
    features.update(llm_features)
    return features
