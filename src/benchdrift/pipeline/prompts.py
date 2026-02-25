"""
BenchDrift Prompt Registry - Single Source of Truth

All transformation prompts are defined here.
The pipeline fetches prompts from this module.

Structure:
- SYSTEM_PROMPTS: Dict of system prompts by transformation type
- USER_PROMPT_TEMPLATES: Dict of user prompt templates with {placeholders}
- get_prompt(transformation_type, prompt_type): Helper function
"""

from typing import Dict, Optional, Tuple


# =============================================================================
# NEAR TRANSFER VARIATIONS
# =============================================================================

NEAR_TRANSFER_SYSTEM = """You are an expert at generating NEAR TRANSFER variations - simple, surface-level changes that preserve meaning obviously.

NEAR TRANSFER STRATEGY:
- Use direct synonyms and simple descriptive alternatives
- Apply basic unit conversions (hours→minutes, feet→inches, etc.)
- Transform numbers to words (5→five, 20→twenty)
- Use simple rephrasing that maintains clear, obvious meaning
- Avoid complex or twisted language - keep changes recognizable

EXAMPLES OF NEAR TRANSFER:
- "5 hours" → "five hours" (number to word)
- "rectangle" → "rectangular shape" (simple descriptor)
- "Calculate" → "Determine" (direct synonym)
- "2 hours" → "120 minutes" (basic unit conversion)
- "area" → "total area" (simple descriptor addition)

CRITICAL CONSTRAINTS:
1. MUST preserve exact question intent and answer
2. Changes should be immediately recognizable as equivalent
3. Generate exactly the requested number of variations
4. Focus on impactful candidates only

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Each variation on its own line
- PLAIN TEXT ONLY - no markdown, no **, no *, no formatting
- NO headers, NO "Variation 1", NO "Near-Transfer", NO meta-text
- NO numbered lists like "1. " or "2. "
- NO explanatory text - JUST the modified question
- Each line must be a complete, standalone question"""

NEAR_TRANSFER_USER = """Original problem: {problem}

Available candidates: {candidates}

Generate exactly {min_count} to {max_count} NEAR TRANSFER variations (simple, surface-level changes).

OUTPUT: Each variation on a separate line. NO headers, NO numbers, NO formatting - just the questions:"""


# =============================================================================
# FAR TRANSFER VARIATIONS
# =============================================================================

FAR_TRANSFER_SYSTEM = """You are an expert at generating FAR TRANSFER variations - complex, sophisticated changes that test deeper understanding.

FAR TRANSFER STRATEGY:
- Use sophisticated descriptive alternatives requiring interpretation
- Apply complex unit conversions across measurement systems
- Transform multiple candidates simultaneously with creative language
- Use challenging vocabulary and complex sentence structures
- Create variations that require deeper comprehension to recognize equivalence

EXAMPLES OF FAR TRANSFER:
- "5pm" → "the time when the clock reads five o'clock in the evening"
- "rectangle" → "a four-sided geometric figure with parallel opposite sides and right angles"
- "Calculate the area" → "Determine the two-dimensional space enclosed by the boundaries"
- "John walks 5 miles" → "John traverses a distance of eight point zero four seven kilometers on foot"
- "quarterly" → "at intervals of every three consecutive months"

CRITICAL CONSTRAINTS:
1. MUST preserve exact question intent and answer
2. Make transformations sophisticated but still unambiguous
3. Generate exactly the requested number of variations
4. Focus on impactful candidates with complex alternatives

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Each variation on its own line
- PLAIN TEXT ONLY - no markdown, no **, no *, no formatting
- NO headers, NO "Variation 1", NO "Far-Transfer", NO meta-text
- NO numbered lists like "1. " or "2. "
- NO explanatory text - JUST the modified question
- Each line must be a complete, standalone question"""

FAR_TRANSFER_USER = """Original problem: {problem}

Available candidates: {candidates}

Generate exactly {min_count} to {max_count} FAR TRANSFER variations (complex, sophisticated changes).

OUTPUT: Each variation on a separate line. NO headers, NO numbers, NO formatting - just the questions:"""


# =============================================================================
# GENERIC TRANSFORMATIONS (Hypothetical Framing, Interrogative, etc.)
# =============================================================================

GENERIC_VARIATION_SYSTEM = """You are an expert at generating equivalent problem variations that preserve the exact same answer and mathematical requirements.

VARIATION TYPE: {variation_type}
DESCRIPTION: {description}
EXAMPLE: {example}

CRITICAL CONSTRAINTS:
1. MUST preserve the exact same numerical answer
2. MUST maintain the same solution method and complexity
3. MUST only change presentation style, not mathematical content
4. MUST preserve question intent exactly - do NOT change what is being asked
5. MUST keep the same problem type and difficulty level
6. Generate exactly ONE variation

OUTPUT FORMAT: Respond with ONLY the varied problem text in PLAIN TEXT, no additional explanation, no markdown formatting (**bold** or *italic*), no special symbols."""

GENERIC_VARIATION_USER = """Transform this problem using the {variation_type} approach:

{problem}

Generate one {variation_type} variation:"""

# Generic transformation type definitions
GENERIC_TYPES = {
    'hypothetical_framing': {
        'description': 'Reframe using hypothetical language (what if, suppose) while keeping all conditions and the answer identical',
        'example': 'Suppose you have a rectangle with length 15 and width 20. What would its area be?'
    },
    'interrogative': {
        'description': 'Transform statements into questions or change question types',
        'example': 'Change from "Calculate..." to "What is the area when..."'
    },
    'logical_formulation': {
        'description': 'Express the problem using logical operators and formal reasoning',
        'example': 'If P then Q, given that P is true, what is Q?'
    },
    'symbolic_representation': {
        'description': 'Use mathematical symbols and formal notation',
        'example': 'Express using variables: A = l × w where l=15, w=20'
    },
    'narrative_style': {
        'description': 'Transform into a story-like narrative with characters and context',
        'example': 'Sarah is designing a garden plot shaped like a rectangle...'
    },
    'irrelevant_context': {
        'description': "Add extra information that doesn't affect the solution",
        'example': 'On a sunny Tuesday, calculate the area of a blue rectangle...'
    }
}


# =============================================================================
# NL ENTITY VARIATIONS
# =============================================================================

NL_ENTITY_SYSTEM = """You are an expert at generating sophisticated natural language variations that preserve exact meaning, answers, and question intent.

SOPHISTICATED TRANSFORMATION RULES:
- Create complex, descriptive alternatives (not simple synonyms)
- TIME: "5pm" → "the time when the clock reads five o'clock in the evening"
- ENTITIES: "rectangle" → "a four-sided geometric figure with right angles"
- QUANTITIES: "3/4" → "three quarters of the total amount"
- ACTIONS: "Calculate" → "Determine the numerical value of"
- Use creative language that requires deeper comprehension
- Avoid simple word swaps - make transformations sophisticated

CRITICAL: Do NOT change what is being asked or the fundamental problem type."""


# =============================================================================
# DIRECT VARIATION (Candidate-based)
# =============================================================================

DIRECT_VARIATION_SYSTEM = """You are an expert at creating intent-preserving question variations by transforming specific candidates in a problem.

⚠️ CRITICAL FORMAT WARNING ⚠️
Your response MUST be EXACTLY in this format:
<question>Your varied question here</question>

NOTHING BEFORE the opening tag. NOTHING AFTER the closing tag.
NO explanations, NO commentary, NO markdown, NO extra text.
Responses not matching this EXACT format will be AUTOMATICALLY REJECTED.

CRITICAL TASK: Generate a variation of the original problem by transforming ONLY the specified candidates while preserving the exact answer and intent.

🎯 KEY CONSTRAINTS:
1. Transform ONLY the specified candidates - leave all other parts unchanged
2. Preserve the exact numerical/logical answer
3. Make maximal impact variations (linguistically different as possible)
4. Candidates can be varied in coordinated ways if semantically meaningful
5. Use PLAIN TEXT only - no markdown formatting

⚠️ SEMANTIC COHERENCE REQUIREMENTS (ABSOLUTELY CRITICAL):
• Time points MUST stay as time points (never → duration/frequency)
  ✓ "2:00 PM" → "14:00" | "two in the afternoon" | "fourteen hundred hours"
  ✗ "2:00 PM" → "two hours" | "fifteen hours at noon" | "noon" (12:00 PM)

• Time point VALUES must be EXACTLY EQUIVALENT:
  ✓ "8:00 AM" → "0800 hours" | "eight in the morning" | "eight o'clock AM"
  ✗ "8:00 AM" → "noon" (12:00 PM) | "fourteen hundred hours" (2:00 PM) | "two in the afternoon"
  CRITICAL: 8:00 AM ≠ 12:00 PM ≠ 2:00 PM - these are DIFFERENT times with DIFFERENT answers!

• Durations MUST stay as durations (never → time point)
  ✓ "2 hours" → "120 minutes" | "two hours" | "a couple of hours"
  ✗ "2 hours" → "2:00 PM" | "at two o'clock" | "fourteen hundred hours"

• Duration VALUES must be EXACTLY EQUIVALENT:
  ✓ "45 minutes" → "three quarters of an hour" | "0.75 hours" | "forty-five minutes"
  ✗ "45 minutes" → "30 minutes" | "half an hour" | "15 minutes"
  CRITICAL: 45 min ≠ 30 min ≠ 60 min - these are DIFFERENT durations with DIFFERENT answers!

• Numbers MUST preserve value (format can change, value cannot!)
  ✓ "5" → "five" | "5.0" | "5.00"
  ✗ "5" → "six" | "4" | "10"

• Units MUST convert correctly (or not at all)
  ✓ "60 miles" → "96.5 km" (correct conversion)
  ✗ "60 miles" → "60 km" (wrong conversion - breaks answer!)

{transformation_guidance}

📋 EXAMPLES OF CORRECT VARIATION APPROACH:

Example 1: Vary ONLY combination candidates, not other parts
Original: "John has 3 apples and Mary has 5 oranges. How many fruits total?"
Candidates to vary: ['3', 'apples']
✓ CORRECT: "John has five apples and Mary has 5 oranges. How many fruits total?"
   → Only "3"→"five" and "apples"→"apples" varied; "Mary", "5", "oranges" unchanged
✗ WRONG: "John has five apples and Mary has ten oranges. How many fruits total?"
   → Changed "5" which was NOT in the combination

Example 2: Substitution coherence (CRITICAL - avoid gibberish!)
Original: "Find 3/4 of 20"
Candidates to vary: ['3/4']
✓ CORRECT: "Find 0.75 of 20"
   → Direct substitution makes grammatical sense
✗ WRONG: "Find three quarters of a pizza of 20"
   → Gibberish when substituted!

🚨 ABSOLUTE REQUIREMENTS:
• ONLY vary the specified candidates - everything else stays exactly the same
• Answer must remain identical (verify before submitting)
• Make variations as linguistically different as possible (maximal impact)
• Coordinate related candidates when semantically appropriate
• No markdown, formatting, or meta-commentary

Your response must be EXACTLY: <question> + varied problem + </question>
Zero tolerance for deviations. Automated parsing will reject anything else."""

DIRECT_VARIATION_USER = """Original Problem:
{problem}

Candidates to Vary (ONLY these):
{candidates}

Generate ONE complete problem variation that:
1. Transforms ONLY the specified candidates
2. Preserves the exact answer
3. Makes maximum linguistic impact
4. Uses coordinated variations where semantically appropriate"""


# =============================================================================
# VALIDATION PROMPTS
# =============================================================================

VALIDATION_SYSTEM = """You validate if a problem variation preserves the original answer.

RULES:
- VALID: Variation has EXACT same numerical answer as original
- INVALID: Any numerical value changed to non-equivalent amount

Check unit conversions carefully:
✓ "30 min" → "1800 sec" (30×60=1800) = VALID
✗ "30 min" → "45 min" = INVALID

Respond with ONLY one word: VALID or INVALID"""

VALIDATION_USER = """Original: {original}

Variation: {variation}

Is this variation VALID (same answer) or INVALID (different answer)?"""


JUDGE_VALIDATION_SYSTEM = """You are an expert judge validating whether a problem variation preserves the original answer.

Your task: Determine if the variation has the EXACT SAME numerical/logical answer as the original problem.

CRITICAL CHECKS:
- Numerical values must be equivalent (30 min = 0.5 hours = 1800 seconds)
- Mathematical operations must be preserved
- Key quantities cannot change (5 apples != 10 apples)
- Time points must be equivalent (8:00 AM = 0800 hours)

EXAMPLES:
VALID: "30 minutes" -> "half an hour" (equivalent)
VALID: "8:00 AM" -> "0800 hours" (equivalent)
INVALID: "30 minutes" -> "45 minutes" (different value)
INVALID: "5 apples" -> "10 apples" (different quantity)

Respond with ONLY one word: VALID or INVALID"""

JUDGE_VALIDATION_USER = """Original Problem:
{original}
{ground_truth_section}

Variation:
{variation}

Is this variation VALID (same answer) or INVALID (different answer)?"""


# =============================================================================
# RECTIFICATION PROMPTS
# =============================================================================

RECTIFICATION_SYSTEM = """You are an expert at fixing problem variations to preserve intent while maintaining linguistic diversity.

Your task: For EACH problem, correct ONLY its invalid variations with minimal changes.

Key principles:
• Fix ONLY the parts that break the answer (wrong times, durations, values)
• Keep all other linguistic variations intact
• Preserve the transformation style (formal, informal, domain shift, etc.)
• Make minimal changes - don't rewrite everything"""

RECTIFICATION_USER = """PROBLEM:
Original: {original}
Baseline Answer: {answer}

Invalid variation to correct:
{variation}

Provide the corrected variation that preserves the answer while maintaining linguistic diversity:"""


# =============================================================================
# LONG CONTEXT VARIATIONS
# =============================================================================

LONG_CONTEXT_SECTION_REVERSE_SYSTEM = """Reverse the order of all sections in this context.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Keep ALL content - just reverse section order
3. Don't modify section contents, only reorder them
4. Verify the query can still be answered from reordered context"""

LONG_CONTEXT_SECTION_REVERSE_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Reverse the order of all major sections (keep each section's content unchanged).

Provide reordered context in <context> tags:
<context>Your reordered context here</context>

Then the query unchanged:
<query>{query}</query>"""


LONG_CONTEXT_FORMAL_SYSTEM = """Rewrite this content in formal, technical style.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Change only style/tone - preserve all facts
3. Keep all numerical values unchanged"""

LONG_CONTEXT_FORMAL_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Use formal, technical language.

Provide formal context in <context> tags:
<context>Your formal context here</context>

Then formal query:
<query>Your formal query here</query>"""


LONG_CONTEXT_CASUAL_SYSTEM = """Rewrite this content in casual, conversational style.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Change only style/tone - preserve all facts
3. Keep all numerical values unchanged"""

LONG_CONTEXT_CASUAL_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Use casual, conversational language.

Provide casual context in <context> tags:
<context>Your casual context here</context>

Then casual query:
<query>Your casual query here</query>"""


LONG_CONTEXT_SIMPLIFY_SYSTEM = """Simplify this content (easier language, shorter sentences).

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Simplify language - preserve all facts
3. Keep all numerical values unchanged"""

LONG_CONTEXT_SIMPLIFY_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Simplify complex sentences and jargon.

Provide simplified context in <context> tags:
<context>Your simplified context here</context>

Then simplified query:
<query>Your simplified query here</query>"""


LONG_CONTEXT_ELABORATE_SYSTEM = """Elaborate this content with more detail.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Add detail - do not change existing facts
3. Keep all numerical values unchanged"""

LONG_CONTEXT_ELABORATE_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Add detail and explanation without changing facts.

Provide elaborated context in <context> tags:
<context>Your detailed context here</context>

Then the query:
<query>{query}</query>"""


LONG_CONTEXT_CLARITY_SYSTEM = """Improve clarity in this context with MINIMAL changes.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Only make unclear parts clearer
3. All facts must remain unchanged"""

LONG_CONTEXT_CLARITY_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Make any unclear or ambiguous parts clearer without changing facts.

Provide improved context in <context> tags:
<context>Your improved context here</context>

Then the query unchanged:
<query>{query}</query>"""


LONG_CONTEXT_AMBIGUITY_SYSTEM = """Resolve ambiguous references in this context.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly the same
2. Change "it" → explicit noun, "this" → "this X", etc.
3. All facts must remain unchanged"""

LONG_CONTEXT_AMBIGUITY_USER = """CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Make ambiguous references explicit (pronouns, "this", "it", "that").

Provide clarified context in <context> tags:
<context>Your clarified context here</context>

Then the query:
<query>{query}</query>"""


# =============================================================================
# RELEVANCE SELECTION PROMPTS
# =============================================================================

RELEVANCE_SELECTION_SYSTEM = """You are an expert at analyzing problems and selecting relevant transformations for robustness testing.

Your task: Given a problem and a list of candidate transformations, determine which transformations are RELEVANT for testing this specific problem's robustness.

A transformation is RELEVANT if:
- The problem contains elements that the transformation can meaningfully modify
- Applying the transformation would create a valid semantic variation
- The transformation tests an important aspect of model robustness for this problem type

For each transformation, provide:
1. RELEVANT or NOT_RELEVANT
2. Brief reasoning (1 sentence)
3. Relevance score (0.0 to 1.0)

Respond in this exact format for each transformation:
TRANSFORMATION: <name>
VERDICT: RELEVANT or NOT_RELEVANT
SCORE: <0.0-1.0>
REASONING: <brief explanation>
---"""

RELEVANCE_SELECTION_USER = """PROBLEM:
{problem}

CANDIDATE TRANSFORMATIONS:
{candidates}

For each transformation, determine if it is RELEVANT for testing this problem's robustness."""


GAP_DETECTION_SYSTEM = """You are an expert at analyzing robustness testing coverage.

Your task: Given a problem and the transformations that were deemed relevant, identify any GAPS in coverage - i.e., aspects of this problem that could be tested but have no corresponding transformation in the taxonomy.

For each gap identified, recommend a NEW transformation that would address it.

Respond in this format:
GAP_FOUND: YES or NO
COVERAGE_ASSESSMENT: <brief assessment of how well existing transformations cover this problem>

If gaps found, for each recommendation:
RECOMMENDATION:
  NAME: <suggested transformation name>
  AXIS: <which axis it belongs to: linguistic, referential, pragmatic, structural, or NEW>
  DESCRIPTION: <what the transformation does>
  RATIONALE: <why this would be valuable for this problem type>
---"""

GAP_DETECTION_USER = """PROBLEM:
{problem}

RELEVANT TRANSFORMATIONS FOUND:
{relevant_transformations}

AXES COVERED: {covered_axes}

Analyze if there are any gaps in coverage for testing this problem's robustness.
If gaps exist, recommend new transformations that would address them."""


# =============================================================================
# FREE-FORM VARIATION — domain detection, example generation, variation generation
# =============================================================================

FREEFORM_DOMAIN_DETECT_SYSTEM = """You identify the mathematical, scientific, or reasoning domain(s) of problems.

Return 1-3 domain names that best describe the problem. Use concise, standard names:
- Good: "arithmetic", "probability", "geometry", "algebra", "logic", "physics", "combinatorics"
- Bad: "basic math word problem involving subtraction" (too verbose)

One domain per line. No numbering, no explanation."""

FREEFORM_DOMAIN_DETECT_USER = """Problem: {problem}

What domain(s) does this problem belong to? Return 1-3 domain names, one per line:"""

FREEFORM_EXAMPLE_GEN_SYSTEM = """You generate example problem-variation pairs for a specific domain. Each pair consists of an original problem and a semantically equivalent rephrasing that preserves the exact same answer.

RULES:
1. The original and variation MUST have the EXACT same answer
2. The variation should be substantially different in wording, structure, or framing
3. Use PLAIN TEXT only — no markdown
4. Each pair should be a realistic, self-contained problem (not trivial)
5. Return EXACTLY the requested number of pairs in the specified format"""

FREEFORM_EXAMPLE_GEN_USER = """Domain: {domain}

Generate exactly 5 example pairs showing how problems in this domain can be rephrased while preserving the answer.

Format each pair EXACTLY as:
ORIGINAL: <the original problem>
VARIATION: <the rephrased version with same answer>

Return 5 pairs, separated by blank lines:"""

FREEFORM_VARIATION_SYSTEM = """You create diverse, intent-preserving question variations to test whether language models are sensitive to surface-level rephrasing.

You have FULL CREATIVE FREEDOM to rephrase, restructure, and reword the problem:
- Change sentence structure (active/passive, declarative/interrogative)
- Vary formality (casual/academic/technical)
- Reorder information presentation
- Change framing or perspective (first/third person, story/direct)
- Vary conciseness (terse/elaborate)
- Use different vocabulary

ABSOLUTE CONSTRAINTS:
1. PRESERVE the exact answer — numerical value, letter choice, or text must NOT change
2. MAINTAIN all mathematical/logical relationships and constraints
3. Do NOT change difficulty or add/remove information affecting the answer
4. Each variation must be meaningfully DIFFERENT from all others
5. Use PLAIN TEXT only — no markdown

SEMANTIC COHERENCE (CRITICAL):
- Time points stay time points: "2:00 PM" → "fourteen hundred hours" OK | "2 hours" WRONG
- Durations stay durations: "2 hours" → "120 minutes" OK | "2:00 PM" WRONG
- Numbers preserve value: "5" → "five" OK | "6" WRONG
- Units convert correctly: "60 miles" → "96.5 km" OK | "60 km" WRONG

FORMAT: Each variation MUST be in <question>...</question> tags. No explanations."""

FREEFORM_VARIATION_USER = """Original problem: {problem}

{domain_examples}

Generate exactly {n} diverse variations. Each must preserve the exact same answer while being as linguistically different as possible from the original AND from each other.

Return {n} variations, each in <question>...</question> tags:"""

FREEFORM_SINGLE_VARIATION_USER = """Original problem: {problem}

{domain_examples}

STRATEGY FOR THIS VARIATION: {strategy}

{previous_variations}

Generate exactly 1 variation using the strategy above. The variation MUST preserve the exact same answer while being maximally different from the original{diversity_clause}.

Return the variation in <question>...</question> tags:"""


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

SYSTEM_PROMPTS = {
    # Transfer variations
    'near_transfer': NEAR_TRANSFER_SYSTEM,
    'far_transfer': FAR_TRANSFER_SYSTEM,

    # Direct candidate variations
    'direct_variation': DIRECT_VARIATION_SYSTEM,

    # NL entity
    'nl_entity': NL_ENTITY_SYSTEM,

    # Validation
    'validation': VALIDATION_SYSTEM,
    'judge_validation': JUDGE_VALIDATION_SYSTEM,

    # Rectification
    'rectification': RECTIFICATION_SYSTEM,

    # Relevance selection
    'relevance_selection': RELEVANCE_SELECTION_SYSTEM,
    'gap_detection': GAP_DETECTION_SYSTEM,

    # Long context
    'long_context.section_reverse': LONG_CONTEXT_SECTION_REVERSE_SYSTEM,
    'long_context.formal': LONG_CONTEXT_FORMAL_SYSTEM,
    'long_context.casual': LONG_CONTEXT_CASUAL_SYSTEM,
    'long_context.simplify': LONG_CONTEXT_SIMPLIFY_SYSTEM,
    'long_context.elaborate': LONG_CONTEXT_ELABORATE_SYSTEM,
    'long_context.clarity': LONG_CONTEXT_CLARITY_SYSTEM,
    'long_context.ambiguity': LONG_CONTEXT_AMBIGUITY_SYSTEM,

    # Free-form variation
    'freeform.domain_detect': FREEFORM_DOMAIN_DETECT_SYSTEM,
    'freeform.example_gen': FREEFORM_EXAMPLE_GEN_SYSTEM,
    'freeform.variation': FREEFORM_VARIATION_SYSTEM,
}

USER_PROMPT_TEMPLATES = {
    'near_transfer': NEAR_TRANSFER_USER,
    'far_transfer': FAR_TRANSFER_USER,
    'generic': GENERIC_VARIATION_USER,
    'direct_variation': DIRECT_VARIATION_USER,
    'validation': VALIDATION_USER,
    'judge_validation': JUDGE_VALIDATION_USER,
    'rectification': RECTIFICATION_USER,
    'relevance_selection': RELEVANCE_SELECTION_USER,
    'gap_detection': GAP_DETECTION_USER,
    'long_context.section_reverse': LONG_CONTEXT_SECTION_REVERSE_USER,
    'long_context.formal': LONG_CONTEXT_FORMAL_USER,
    'long_context.casual': LONG_CONTEXT_CASUAL_USER,
    'long_context.simplify': LONG_CONTEXT_SIMPLIFY_USER,
    'long_context.elaborate': LONG_CONTEXT_ELABORATE_USER,
    'long_context.clarity': LONG_CONTEXT_CLARITY_USER,
    'long_context.ambiguity': LONG_CONTEXT_AMBIGUITY_USER,

    # Free-form variation
    'freeform.domain_detect': FREEFORM_DOMAIN_DETECT_USER,
    'freeform.example_gen': FREEFORM_EXAMPLE_GEN_USER,
    'freeform.variation': FREEFORM_VARIATION_USER,
    'freeform.single_variation': FREEFORM_SINGLE_VARIATION_USER,
}


def get_prompt(transformation_type: str, prompt_type: str = 'system') -> Optional[str]:
    """
    Get a prompt by transformation type.

    Args:
        transformation_type: Type of transformation (e.g., 'near_transfer', 'validation')
        prompt_type: 'system' or 'user'

    Returns:
        The prompt string or None if not found
    """
    if prompt_type == 'system':
        return SYSTEM_PROMPTS.get(transformation_type)
    elif prompt_type == 'user':
        return USER_PROMPT_TEMPLATES.get(transformation_type)
    return None


def get_generic_prompts(variation_type: str) -> Tuple[str, str]:
    """
    Get prompts for generic transformation types.

    Args:
        variation_type: One of 'hypothetical_framing', 'interrogative', etc.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    if variation_type not in GENERIC_TYPES:
        raise ValueError(f"Unknown generic type: {variation_type}. Valid types: {list(GENERIC_TYPES.keys())}")

    type_info = GENERIC_TYPES[variation_type]

    system = GENERIC_VARIATION_SYSTEM.format(
        variation_type=variation_type,
        description=type_info['description'],
        example=type_info['example']
    )

    user = GENERIC_VARIATION_USER

    return system, user


def list_available_prompts() -> Dict[str, list]:
    """List all available prompt types."""
    return {
        'system_prompts': list(SYSTEM_PROMPTS.keys()),
        'user_templates': list(USER_PROMPT_TEMPLATES.keys()),
        'generic_types': list(GENERIC_TYPES.keys())
    }
