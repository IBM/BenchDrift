#!/usr/bin/env python3
"""
Long Context Variations - Focused set of variations for long-context prompts.

Assumes input format: Context (multiple lines) + Query (last line)
Preserves answer while varying structure, formatting, and clarity.
"""

import re
from typing import Dict, List, Tuple



class LongContextStructure:
    """Parsed structure of long context prompt."""

    def __init__(self, full_text: str):
        self.full_text = full_text
        self.lines = full_text.split('\n')

        # Assumption: Last non-empty line = query, rest = context
        non_empty = [i for i, line in enumerate(self.lines) if line.strip()]
        if non_empty:
            self.query_line_idx = non_empty[-1]
            self.query = self.lines[self.query_line_idx].strip()
            self.context_lines = self.lines[:self.query_line_idx]
            self.context = '\n'.join(self.context_lines).strip()
        else:
            self.query = ""
            self.context = full_text
            self.query_line_idx = len(self.lines)
            self.context_lines = self.lines

        # Detect paragraphs (double newline separated)
        self.paragraphs = [p.strip() for p in self.context.split('\n\n') if p.strip()]
        self.has_paragraphs = len(self.paragraphs) > 1

        # Detect sections (lines starting with UPPERCASE_WORD:)
        self.sections = self._detect_sections()

    def _detect_sections(self) -> List[Dict]:
        """Detect sections like RULES:, EXAMPLES:, etc."""
        sections = []
        current_section = None
        current_content = []

        for line in self.context_lines:
            # Check if line is a section header (UPPERCASE_WORD:)
            match = re.match(r'^([A-Z][A-Z_\s]+):\s*$', line.strip())
            if match:
                # Save previous section
                if current_section:
                    sections.append({
                        'heading': current_section,
                        'content': '\n'.join(current_content),
                        'lines': current_content[:]
                    })
                # Start new section
                current_section = match.group(1)
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            sections.append({
                'heading': current_section,
                'content': '\n'.join(current_content),
                'lines': current_content[:]
            })

        return sections


def get_long_context_transformation_types():
    """
    Get focused set of long context transformations.

    Only most relevant variations from README + analysis.
    Returns dict with prompts and metadata.
    """
    return {
        # ===== DETERMINISTIC VARIATIONS (Format & Surface) =====
        'long_context.format.quotes': {
            'type': 'deterministic',
            'variants': [
                {
                    'name': 'single_to_double',
                    'function': lambda text: text.replace("'", '"'),
                    'capability': 'quote_robustness',
                },
                {
                    'name': 'double_to_single',
                    'function': lambda text: text.replace('"', "'"),
                    'capability': 'quote_robustness',
                },
            ]
        },
        'long_context.format.whitespace': {
            'type': 'deterministic',
            'variants': [
                {
                    'name': 'add_extra_spaces',
                    'function': lambda text: text.replace(' ', '  '),  # Double spaces
                    'capability': 'whitespace_robustness',
                },
                {
                    'name': 'normalize_whitespace',
                    'function': lambda text: ' '.join(text.split()),  # Normalize to single spaces
                    'capability': 'whitespace_robustness',
                },
            ]
        },
        'long_context.format.case': {
            'type': 'deterministic',
            'variants': [
                {
                    'name': 'uppercase_headers',
                    'function': lambda text: '\n'.join(
                        line.upper() if line.strip().endswith(':') else line
                        for line in text.split('\n')
                    ),
                    'capability': 'case_sensitivity',
                },
                {
                    'name': 'lowercase_headers',
                    'function': lambda text: '\n'.join(
                        line.lower() if line.strip().endswith(':') else line
                        for line in text.split('\n')
                    ),
                    'capability': 'case_sensitivity',
                },
            ]
        },

        # ===== LLM-BASED VARIATIONS (Structure & Positioning) =====
        'long_context.positioning.sections': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'reverse_all',
                    'prompt': """Reverse the order of all sections in this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Reverse the order of all major sections (keep each section's content unchanged).

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. All facts, numbers, and logic NEEDED TO DERIVE {answer} must remain unchanged
3. Verify that {query} can still be answered with {answer} from your reordered context
4. Don't modify section contents, only reorder them
5. Keep ALL content - just reverse section order

Provide reordered context in <context> tags:
<context>Your reordered context here</context>

Then the query unchanged:
<query>{query}</query>""",
                    'capability': 'order_robustness',
                },
                {
                    'name': 'examples_first',
                    'prompt': """Move all examples/demonstrations to the beginning of the context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Identify example sections and move them to the start. Keep other sections in same relative order.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Keep ALL content - just reorder sections
3. Preserve all facts and numerical values

Provide reordered context in <context> tags:
<context>Your reordered context here</context>

Then the query unchanged:
<query>{query}</query>""",
                    'capability': 'order_robustness',
                },
            ]
        },
        'long_context.positioning.paragraphs': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'reverse',
                    'prompt': """Reverse the order of paragraphs in this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Reverse paragraph order (last paragraph becomes first, etc.). Keep each paragraph's content unchanged.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Keep ALL content - just reverse paragraph order
3. Preserve all facts and numerical values

Provide reordered context in <context> tags:
<context>Your reordered context here</context>

Then the query unchanged:
<query>{query}</query>""",
                    'capability': 'order_robustness',
                },
            ]
        },
        'long_context.content.removal': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'drop_examples',
                    'prompt': """Remove example sections while keeping essential information:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Remove example/demonstration sections. Keep rules, definitions, and essential info.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Remove only redundant examples - keep essential info
3. Preserve all facts needed to answer the query

Provide condensed context in <context> tags:
<context>Your condensed context here</context>

Then the query unchanged:
<query>{query}</query>""",
                    'capability': 'information_filtering',
                },
            ]
        },

        # ===== LLM-BASED VARIATIONS (Quality Improvements) =====
        'long_context.quality.clarity': {
            'type': 'llm',
            'prompt': """Improve clarity in this context with MINIMAL changes:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Make any unclear or ambiguous parts clearer without changing facts.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. All facts, numbers, and logic NEEDED TO DERIVE {answer} must remain unchanged
3. After clarification, verify that {query} still produces {answer}
4. Only make unclear parts clearer
5. Test: Can a solver still derive {answer} from your clarified context?

Provide ONLY the improved context wrapped in <context> tags:
<context>Your improved context here</context>

Then the query unchanged:
<query>{query}</query>""",
            'capability': 'clarity_robustness',
        },

        'long_context.quality.completeness': {
            'type': 'llm',
            'prompt': """Add any missing implied information to this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Add information that is implied but not explicitly stated.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Add ONLY missing information that supports deriving {answer}
3. DO NOT change any existing numbers, formulas, or logic
4. After adding information, verify {query} still yields {answer}
5. Test: Does your enhanced context still produce the same {answer}?

Provide the complete context with additions in <context> tags:
<context>Your enhanced context here</context>

Then the query unchanged:
<query>{query}</query>""",
            'capability': 'inference_robustness',
        },

        'long_context.quality.ambiguity': {
            'type': 'llm',
            'prompt': """Resolve ambiguous references in this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Make ambiguous references explicit (pronouns, "this", "it", "that").

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Change "it" → "the tool", "this" → "this parameter", etc.
3. All facts, numbers, and logic NEEDED TO DERIVE {answer} must remain unchanged
4. Only resolve ambiguous references
5. Verify {query} still produces {answer} after changes

Provide the context with explicit references in <context> tags:
<context>Your clarified context here</context>

Then the query:
<query>{query}</query>""",
            'capability': 'coreference_robustness',
        },

        # ===== LLM-BASED VARIATIONS (Restyling) =====
        'long_context.style.redundancy': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'remove',
                    'prompt': """Remove redundant information from this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Remove repetitive/redundant information while preserving all unique facts.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Keep all unique facts - remove only redundancy
3. Preserve numerical values and key information

Provide concise context in <context> tags:
<context>Your concise context here</context>

Then the query:
<query>{query}</query>""",
                    'capability': 'redundancy_sensitivity',
                },
                {
                    'name': 'add',
                    'prompt': """Add clarifying redundancy to this context:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Add helpful redundancy/repetition of key points for clarity.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Add clarifying repetition - don't change facts
3. Preserve all numerical values

Provide enhanced context in <context> tags:
<context>Your enhanced context here</context>

Then the query:
<query>{query}</query>""",
                    'capability': 'redundancy_dependency',
                },
            ]
        },

        'long_context.style.formality': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'formal',
                    'prompt': """Rewrite this context in formal, technical style:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Use formal, technical language.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Change only style/tone - preserve all facts
3. Keep all numerical values unchanged

Provide formal context in <context> tags:
<context>Your formal context here</context>

Then formal query:
<query>Your formal query here</query>""",
                    'capability': 'formality_robustness',
                },
                {
                    'name': 'casual',
                    'prompt': """Rewrite this context in casual, conversational style:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Use casual, conversational language.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Change only style/tone - preserve all facts
3. Keep all numerical values unchanged

Provide casual context in <context> tags:
<context>Your casual context here</context>

Then casual query:
<query>Your casual query here</query>""",
                    'capability': 'formality_robustness',
                },
            ]
        },

        'long_context.style.complexity': {
            'type': 'llm',
            'variants': [
                {
                    'name': 'simplify',
                    'prompt': """Simplify this context (easier language, shorter sentences):

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Simplify complex sentences and jargon.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Simplify language - preserve all facts
3. Keep all numerical values unchanged

Provide simplified context in <context> tags:
<context>Your simplified context here</context>

Then simplified query:
<query>Your simplified query here</query>""",
                    'capability': 'complexity_robustness',
                },
                {
                    'name': 'elaborate',
                    'prompt': """Elaborate this context with more detail:

CONTEXT:
{context}

QUERY:
{query}

ANSWER (MUST remain):
{answer}

TASK: Add detail and explanation without changing facts.

CRITICAL CONSTRAINTS:
1. The answer MUST stay exactly: {answer}
2. Add detail - don't change existing facts
3. Keep all numerical values unchanged

Provide elaborated context in <context> tags:
<context>Your detailed context here</context>

Then the query:
<query>{query}</query>""",
                    'capability': 'complexity_robustness',
                },
            ]
        },
    }


def is_long_context(text: str, min_length: int = 500) -> bool:
    """Check if text qualifies as long context."""
    return len(text) >= min_length
