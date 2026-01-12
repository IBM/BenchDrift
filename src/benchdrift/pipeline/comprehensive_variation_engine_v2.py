"""
Comprehensive Benchmark Variation Engine
Applies systematic local variations to benchmark problems for model drift measurement.
Uses spaCy for NL entity detection + Pattern matching for math topic detection.
"""

import json
import re
# spaCy import moved to conditional loading to avoid GPU conflicts
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from pathlib import Path
import argparse
from dateutil import parser as date_parser
from datetime import datetime, timedelta
import sympy as sp
from pint import UnitRegistry
import warnings

warnings.filterwarnings('ignore')

# Enhanced consistency checking
from collections import Counter

# Logging setup
import logging
logger = logging.getLogger('BenchDrift.VariationEngine')


# ============================================================================
# COMPOSITE DETECTION RULES - Data-Driven from Actual Problem Patterns
# ============================================================================

MATH_COMPOSITE_RULES = {
    'operators': ['+', '-', '*', '/', '=', '×', '÷', '(', ')', '%'],
    'inequalities': ['<', '>', '≤', '≥', '≠', '≈'],  # NEW: From MMLU/GSM8K
    'special_symbols': [':', '|', '.', ',', '^', '√', '∛', 'π', 'e'],  # NEW: Ratios, abs value, decimals, cube root, euler
    'money_symbols': ['$', '¢', '€', '£', '¥'],  # NEW: From GSM8K (very common!)
    'pos_tags': ['NUM', 'SYM'],
    'dependencies': ['compound', 'nummod', 'conj', 'cc', 'neg'],
    'connecting_words': ['of', 'to', 'per', 'times', 'sqrt', 'root'],  # NEW: "60% of", "$2 per", "3 times", "square root"
    'percentage_patterns': ['%', 'percent', 'percentage'],  # NEW: Percentage composites
    'fraction_words': ['half', 'third', 'quarter', 'thirds', 'quarters'],  # NEW: "a third of"
    'rate_indicators': ['per', 'each', 'every'],  # NEW: Rate composites
    # NEW: LaTeX patterns (from Math-Verify) - Critical for MMLU
    'latex_patterns': [
        r'\$\$[^\$]+\$\$',                      # Display math: $$...$$
        r'\$[^\$]+\$',                          # Inline math: $...$
        r'\\boxed\{[^\}]+\}',                   # Boxed answers: \boxed{...}
        r'\\\[[^\]]+\\\]',                      # Display brackets: \[...\]
        r'\\frac\{[^\}]+\}\{[^\}]+\}',         # Fractions: \frac{a}{b}
        r'\\sqrt(?:\[[^\]]+\])?\{[^\}]+\}',    # Roots: \sqrt{x} or \sqrt[3]{x}
        r'\\[a-zA-Z]+\{[^\}]+\}',              # Other LaTeX commands: \command{...}
    ],
}

TEMPORAL_COMPOSITE_RULES = {
    'time_units': ['hour', 'minute', 'second', 'day', 'week', 'month', 'year'],
    'time_units_abbrev': ['hr', 'min', 'sec', 'am', 'pm', 'a.m.', 'p.m.'],
    'eras': ['BC', 'AD', 'BCE', 'CE'],  # NEW: From TOT dataset (critical!)
    'seasons': ['spring', 'summer', 'fall', 'autumn', 'winter'],  # NEW
    'named_times': ['noon', 'midnight', 'dawn', 'dusk', 'evening', 'morning'],  # NEW
    'relative_time': ['ago', 'later', 'earlier', 'before', 'after', 'since', 'until', 'yesterday', 'today', 'tomorrow'],  # NEW
    'time_phrases': ['quarter', 'half', "o'clock", 'past', 'to'],  # NEW: "quarter past 3"
    'larger_units': ['fortnight', 'decade', 'century', 'millennium'],  # NEW
    'pos_tags': ['NUM', 'NOUN'],
    'dependencies': ['compound', 'nummod', 'conj', 'cc', 'amod', 'appos', 'punct'],
    'separators': [':', '/', '-'],  # Time/date separators
}

NL_COMPOSITE_RULES = {
    'pos_tags': ['PROPN', 'NOUN'],
    'dependencies': [
        'compound', 'conj', 'cc', 'flat', 'amod', 'nmod', 'appos',
        'prt',      # NEW: Phrasal verbs ("pick up")
        'prep',     # NEW: Prepositions (for "on the table")
        'pobj',     # NEW: Prepositional object
        'quantmod'  # NEW: Quantifier modifier
    ],
    'coordinators': ['and', 'or', ',', '&'],
    'titles': ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Professor'],  # NEW
    'multi_word_indicators': ['a', 'lot', 'of', 'bunch'],  # NEW: "a lot of"
    'multi_word_patterns': [  # NEW: Common multi-word patterns as lists
        ['a', 'lot', 'of'],
        ['a', 'few'],
        ['a', 'bunch', 'of'],
    ],
}


def detect_latex_composites(problem: str) -> List[Tuple[str, int, int]]:
    r"""
    Detect LaTeX expressions as atomic composites (from Math-Verify approach).
    Critical for MMLU problems like: "What are the dimensions of $\hat{u}^t \hat{u}$?"

    Args:
        problem: Problem text that may contain LaTeX

    Returns:
        List of (latex_expression, start_pos, end_pos) tuples
    """
    composites = []

    for pattern in MATH_COMPOSITE_RULES['latex_patterns']:
        for match in re.finditer(pattern, problem):
            composites.append((match.group(0), match.start(), match.end()))

    return composites


def normalize_latex_for_sympy(latex_str: str) -> str:
    """
    Normalize malformed LaTeX operators for robust SymPy parsing (from Math-Verify).
    Handles common issues like missing braces, malformed commands, etc.

    Args:
        latex_str: LaTeX string that may be malformed

    Returns:
        Normalized string suitable for SymPy parsing
    """
    normalized = latex_str

    # Remove dollar signs if present
    normalized = normalized.replace('$$', '').replace('$', '')

    # Remove common LaTeX formatting commands that don't affect value (do this early)
    for cmd in ['\\mathrm', '\\displaystyle', '\\text', '\\mathit', '\\mathbf']:
        normalized = normalized.replace(cmd, '')

    # Handle malformed \frac (missing braces)
    # \frac34 → \frac{3}{4}
    normalized = re.sub(r'\\frac\s*(\d+)\s*(\d+)', r'(\1/\2)', normalized)

    # Handle malformed \sqrt (missing braces)
    # \sqrt16 → sqrt(16)
    normalized = re.sub(r'\\sqrt\s*(\d+)', r'sqrt(\1)', normalized)

    # Handle well-formed \frac{a}{b} → (a/b)
    normalized = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'(\1/\2)', normalized)

    # Handle well-formed \sqrt{x} → sqrt(x)
    normalized = re.sub(r'\\sqrt\{([^\}]+)\}', r'sqrt(\1)', normalized)

    # Replace LaTeX operators with SymPy equivalents
    latex_to_sympy_ops = {
        '\\cdot': '*',
        '\\times': '*',
        '\\div': '/',
        '\\pm': '+-',  # Plus-minus (approximate)
    }

    for latex_op, sympy_op in latex_to_sympy_ops.items():
        normalized = normalized.replace(latex_op, sympy_op)

    # Clean up any remaining braces
    normalized = normalized.replace('{', '').replace('}', '')

    return normalized.strip()


# ============================================================================
# FRAGMENT-BASED CONTEXT DETECTION
# ============================================================================

class Fragment:
    """
    Semantic context boundary that groups related entities.
    Fragments contain composites, which contain atomic candidates.

    Hierarchy: Fragment → Composites → Atomic Candidates

    Example:
        Fragment("3 apples for $5", type="transaction")
        ├─ Composite("3 apples")
        │   ├─ Atomic("3")
        │   └─ Atomic("apples")
        └─ Composite("$5")
    """

    _id_counter = 0

    def __init__(self, text: str, span: Tuple[int, int], fragment_type: str):
        self.id = Fragment._id_counter
        Fragment._id_counter += 1

        self.text = text
        self.span = span  # (start, end) in original problem
        self.type = fragment_type  # 'transaction', 'math_expression', 'temporal_range', etc.

        # Will be populated by existing composite detection
        self.composites = []  # List of composite expressions within this fragment
        self.atomic_candidates = []  # List of atomic candidates

        # Relationships between entities in this fragment
        self.internal_relationships = {}  # e.g., {'rate': {'item': '3 apples', 'price': '$5'}}

        # Metadata for combination selection
        self.priority = 1.0  # Higher for fragments with strong internal dependencies

    def __repr__(self):
        return f"Fragment(id={self.id}, type={self.type}, text='{self.text[:30]}...')"

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'text': self.text,
            'span': self.span,
            'type': self.type,
            'composites': [c if isinstance(c, str) else str(c) for c in self.composites],
            'relationships': self.internal_relationships,
            'priority': self.priority
        }


# Fragment type patterns - identifies semantic context boundaries
FRAGMENT_PATTERNS = {
    'rate_expression': [
        # "60 miles per hour", "5 dollars per item"
        r'(\d+(?:\.\d+)?)\s+(\w+)\s+per\s+(\w+)',
        # "60 miles in 2 hours", "5 dollars in 2 items"
        r'(\d+(?:\.\d+)?)\s+(\w+)\s+in\s+(\d+(?:\.\d+)?)\s+(\w+)',
        # "3 apples for $5", "2 items for 10 dollars"
        r'(\d+(?:\.\d+)?)\s+(\w+)\s+for\s+\$?(\d+(?:\.\d+)?)',
    ],
    'temporal_range': [
        # "from 2000 to 2010", "from Monday to Friday"
        r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$|\.|\,)',
        # "between 100 BC and 50 AD" - more permissive for era markers
        r'between\s+([\w\s]+(?:BC|AD|BCE|CE)?)\s+and\s+([\w\s]+(?:BC|AD|BCE|CE)?)',
    ],
    'comparative': [
        # "A is 2 times bigger than B", "twice as many"
        r'(\d+(?:\.\d+)?)\s+times\s+(?:as\s+)?(\w+)',
        r'(twice|thrice|half)\s+as\s+(\w+)',
    ],
    'money_with_unit': [
        # "$25 worth of apples", "$5 per item"
        r'\$(\d+(?:\.\d+)?)\s+(?:worth\s+of|per|for|to)\s+(\w+)',
    ]
}


def detect_fragments_via_patterns(problem: str) -> List[Fragment]:
    """
    Pattern-based fragment detection for known semantic structures.
    These are high-confidence patterns where entities should stay together.
    """
    fragments = []

    for fragment_type, patterns in FRAGMENT_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, problem):
                span = (match.start(), match.end())
                text = match.group(0).strip()

                fragment = Fragment(text, span, fragment_type)

                # Set priority based on type
                if fragment_type in ['rate_expression', 'comparative']:
                    fragment.priority = 2.0  # High priority - strong dependencies
                elif fragment_type == 'temporal_range':
                    fragment.priority = 1.5

                # Extract internal structure
                if fragment_type == 'rate_expression':
                    fragment.internal_relationships = {
                        'relationship': 'rate',
                        'components': list(match.groups())
                    }
                elif fragment_type == 'comparative':
                    fragment.internal_relationships = {
                        'relationship': 'comparative',
                        'components': list(match.groups())
                    }

                fragments.append(fragment)

    return fragments


def detect_math_expression_fragments(problem: str) -> List[Fragment]:
    """
    Detect mathematical expressions as fragments using existing SymPy detection.
    Reuses the existing extract_mathematical_expressions_with_sympy logic.
    """
    fragments = []

    # First, detect LaTeX expressions (they're atomic composites)
    latex_expressions = detect_latex_composites(problem)
    for latex_expr, start, end in latex_expressions:
        fragment = Fragment(latex_expr, (start, end), 'latex_expression')
        fragment.priority = 2.0  # High priority - should stay intact
        fragment.internal_relationships = {'relationship': 'latex', 'atomic': True}
        fragments.append(fragment)

    # Then detect plain math expressions (if SymPy is available)
    try:
        # We'll integrate with MathDomainVariationEngine later
        # For now, use simple pattern matching for common structures

        # Equations: "3x + 5 = 20"
        equation_pattern = r'[a-z0-9\+\-\*/\^\(\)\s]+\s*=\s*[a-z0-9\+\-\*/\^\(\)\s]+'
        for match in re.finditer(equation_pattern, problem):
            text = match.group(0).strip()
            span = (match.start(), match.end())

            # Validate it's actually mathematical
            if any(op in text for op in ['+', '-', '*', '/', '=', '^']):
                fragment = Fragment(text, span, 'math_equation')
                fragment.priority = 1.8  # High priority - maintain equation balance
                fragment.internal_relationships = {'relationship': 'equation'}
                fragments.append(fragment)

    except Exception as e:
        pass  # Silently fail if math detection has issues

    return fragments


def detect_fragments_comprehensive(problem: str) -> List[Fragment]:
    """
    Multi-strategy fragmentation with fallback hierarchy:
    1. Pattern-based (high confidence, fast)
    2. Math expression-based (for equations, LaTeX)
    3. Dependency-based (for general structure) - TODO

    Returns non-overlapping fragments sorted by position.
    """
    all_fragments = []
    covered_spans = []

    # PHASE 1: Pattern-based (highest confidence)
    pattern_fragments = detect_fragments_via_patterns(problem)
    for frag in pattern_fragments:
        if not _overlaps_with_any(frag.span, covered_spans):
            all_fragments.append(frag)
            covered_spans.append(frag.span)

    # PHASE 2: Math expression fragmentation
    math_fragments = detect_math_expression_fragments(problem)
    for frag in math_fragments:
        if not _overlaps_with_any(frag.span, covered_spans):
            all_fragments.append(frag)
            covered_spans.append(frag.span)

    # PHASE 3: TODO - Dependency-based for remaining text
    # This would use spaCy to fragment uncovered text
    # For now, we'll let uncovered text be handled by existing detection

    # Sort by position
    all_fragments.sort(key=lambda f: f.span[0])

    return all_fragments


def _overlaps_with_any(span: Tuple[int, int], covered_spans: List[Tuple[int, int]]) -> bool:
    """Check if span overlaps with any covered span"""
    start, end = span
    for covered_start, covered_end in covered_spans:
        if not (end <= covered_start or start >= covered_end):
            return True
    return False


def enrich_candidates_with_fragments(candidates: List[Dict], fragments: List[Fragment], problem_text: str) -> List[Dict]:
    """
    Enrich candidates with fragment metadata by mapping them to detected fragments.
    Candidates within the same fragment will have the same fragment_id.

    This is a standalone utility function that can be used by any pipeline.
    """
    enriched_candidates = []

    for candidate in candidates:
        cand_enriched = candidate.copy()
        cand_enriched['fragment_id'] = None
        cand_enriched['fragment_type'] = None
        cand_enriched['fragment_priority'] = 1.0

        # Find which fragment (if any) contains this candidate
        cand_text = candidate['text']
        cand_pos = candidate.get('pos', problem_text.find(cand_text))

        for fragment in fragments:
            frag_start, frag_end = fragment.span
            # Check if candidate is within this fragment
            if frag_start <= cand_pos < frag_end:
                cand_enriched['fragment_id'] = fragment.id
                cand_enriched['fragment_type'] = fragment.type
                cand_enriched['fragment_priority'] = fragment.priority
                break  # Use first matching fragment

        enriched_candidates.append(cand_enriched)

    return enriched_candidates


def create_fragment_context_for_prompt(fragments: List[Fragment], candidates_enriched: List[Dict]) -> str:
    """
    Create a concise summary of fragments and their relationships for the model prompt.

    This is a standalone utility function that can be used by any pipeline.
    """
    if not fragments:
        return "No semantic fragments detected. Use standard selection strategy."

    # Group candidates by fragment
    fragment_to_candidates = {}
    for i, cand in enumerate(candidates_enriched):
        frag_id = cand.get('fragment_id')
        if frag_id is not None:
            if frag_id not in fragment_to_candidates:
                fragment_to_candidates[frag_id] = []
            fragment_to_candidates[frag_id].append(i)

    if not fragment_to_candidates:
        return "No semantic fragments detected. Use standard selection strategy."

    # Create summary
    context_lines = [f"Detected {len(fragments)} semantic fragments:"]
    for frag in fragments:
        if frag.id in fragment_to_candidates:
            cand_indices = fragment_to_candidates[frag.id]
            context_lines.append(
                f"  • Fragment {frag.id} ({frag.type}, priority={frag.priority:.1f}): "
                f"'{frag.text[:40]}...' contains candidates {cand_indices}"
            )

    return "\n".join(context_lines)


def clean_model_response(text: str) -> str:
    """
    Clean model-generated text to remove unwanted formatting and ensure plain text.

    Args:
        text: Raw text from model

    Returns:
        Cleaned plain text suitable for questions
    """
    if not text:
        return text
    
    # Remove markdown formatting (more aggressive)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*  
    text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _underline_
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
    
    # Remove ALL remaining formatting symbols
    text = re.sub(r'[*_`#\[\]{}]', '', text)      # Remove stray formatting characters
    text = re.sub(r'---+', '', text)              # Remove horizontal rules
    
    # Remove numbered list prefixes 
    text = re.sub(r'^\d+\.\s*', '', text)         # Remove "1. ", "2. " etc at start
    text = re.sub(r'^\d+\)\s*', '', text)         # Remove "1) ", "2) " etc at start
    
    # Remove section headers and meta text
    unwanted_headers = [
        r'\*\*.*?Variation.*?\*\*',
        r'\*\*.*?Transfer.*?\*\*', 
        r'Variation \d+',
        r'EXAMPLE:.*',
        r'OUTPUT:.*'
    ]
    for pattern in unwanted_headers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)              # Multiple spaces to single space
    text = text.strip()                           # Remove leading/trailing whitespace
    
    # Remove common unwanted prefixes/suffixes that models add
    unwanted_prefixes = [
        "Here is", "Here's", "The answer is", "The question is", "Modified:",
        "Variation:", "Problem:", "Question:", "Here is the", "Here's the",
        "EXAMPLE:", "OUTPUT:", "Format:", "Variation 1", "Variation 2", "Variation 3",
        "Near-Transfer", "Far-Transfer", "Variations", "Total"
    ]
    unwanted_suffixes = [
        "This preserves", "This maintains", "This keeps", "The answer remains"
    ]
    
    # More aggressive prefix removal
    for prefix in unwanted_prefixes:
        # Case insensitive removal
        pattern = re.compile(re.escape(prefix), re.IGNORECASE)
        if pattern.match(text):
            text = pattern.sub('', text, count=1).strip()
            if text.startswith(':'):
                text = text[1:].strip()
    
    for suffix in unwanted_suffixes:
        if text.lower().endswith(suffix.lower()):
            text = text[:-len(suffix)].strip()
    
    # Final validation - if text doesn't look like a proper question, reject it
    if not text or len(text.strip()) < 10:
        return ""
    
    # Check if it's just structural/meta text
    meta_patterns = [
        r'^\d+\s*(total|variations?)',
        r'^---+$',
        r'^\*\*.*\*\*$',
        r'^variation\s*\d*$'
    ]
    
    for pattern in meta_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return ""
    
    return text


def is_valid_question(text: str) -> bool:
    """
    Validate if text looks like a proper question/problem statement.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text appears to be a valid question
    """
    if not text or len(text.strip()) < 20:  # Increase minimum length
        return False
    
    text = text.strip()
    text_lower = text.lower()
    
    # REJECT: Pure meta-content that models often generate
    meta_content_patterns = [
        r'^.{0,5}(variation|example|output|format|near|far|transfer)',  # Starts with these words
        r'^\d+\s*(total|variations?)',                                  # "5 total", "3 variations"  
        r'^---+$',                                                      # Just dashes
        r'^\*\*.*\*\*$',                                               # Just bold text
        r'^(here is|here\'s|the answer is)',                          # Common prefixes
        r'variation\s*\d+',                                            # "Variation 1", "Variation 2"
        r'^\d+\s*variations?\s*\(',                                    # "5 variations (total)"
        r'transfer\s*variations?',                                     # "Transfer variations", "Near transfer"
        r'^\s*\d+\.\s*$',                                             # Just numbered item with no content
        r'^\s*-+\s*$',                                                # Just dashes
        r'^.*total\)?\s*$',                                           # Ends with "total" or "total)"
    ]
    
    for pattern in meta_content_patterns:
        if re.match(pattern, text_lower):
            return False
    
    # REJECT: If text is mostly formatting/structure
    if text.count('**') > 4:  # Too much bold formatting
        return False
    
    if len(text.replace('*', '').replace('-', '').replace('(', '').replace(')', '').strip()) < 15:
        return False  # After removing formatting, too short
    
    # REQUIRE: Must contain actual question/problem content
    question_indicators = [
        'calculate', 'find', 'determine', 'what', 'how', 'if', 'takes', 'requires',
        'minutes', 'hours', 'seconds', 'time', 'will', 'would', 'should', 'is', 'are',
        'paint', 'walk', 'reach', 'complete', 'number', 'area', 'average', 'same pace'
    ]
    
    if not any(indicator in text_lower for indicator in question_indicators):
        return False
    
    # REQUIRE: Should be a complete sentence (has basic sentence structure)
    # Must have some kind of action/verb and reasonable sentence length
    words = text_lower.split()
    if len(words) < 10:  # Too short to be a proper problem statement
        return False
    
    return True


class VariationConsistencyEnhancer:
    """Enhanced consistency methods integrated into the main variation engine"""
    
    def __init__(self, nlp=None):
        self.nlp = nlp
        
        # Time format patterns that can conflict
        self.time_conflict_patterns = [
            (r'(\d{1,2}:\d{2})\s+(AM|PM)\s+(AM|PM)', r'\1 \2'),  # 10:00 AM AM → 10:00 AM
            (r'(\d{1,2}:\d{2}:\d{2})\s+(AM|PM)\s+(AM|PM)', r'\1 \2'),  # 10:00:00 AM AM → 10:00:00 AM
            (r'(\d{1,2}:\d{2})\s+AM\s+AM', r'\1 AM'),  # Specific AM repetition
            (r'(\d{1,2}:\d{2})\s+PM\s+PM', r'\1 PM'),  # Specific PM repetition
        ]
        
        # Advanced libraries for intelligent detection
        self.semantic_detectors = self._initialize_semantic_detectors()
    
    def _initialize_semantic_detectors(self):
        """Initialize advanced semantic detection using libraries"""
        detectors = {}
        
        # spaCy-based entity and dependency detection
        if self.nlp:
            detectors['spacy_ner'] = True
            detectors['spacy_deps'] = True
            detectors['spacy_similarity'] = True
        
        # Try to initialize WordNet for semantic similarity
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            detectors['wordnet'] = wordnet
            logger.debug("✅ WordNet loaded for semantic similarity")
        except:
            detectors['wordnet'] = None
            logger.debug("⚠️ WordNet not available")
        
        # Try to initialize SymPy for mathematical relationships
        try:
            import sympy as sp
            detectors['sympy'] = sp
            logger.debug("✅ SymPy available for mathematical analysis")
        except:
            detectors['sympy'] = None
        
        return detectors
    
    def enhance_variation_consistency(self, original_problem: str, modified_problem: str, 
                                    original_component: str, transformation_type: str) -> str:
        """Main method to enhance a variation for consistency"""
        
        # Step 1: Fix obvious formatting issues
        cleaned_problem = self._fix_formatting_issues(modified_problem)
        
        # Step 2: Fix entity consistency issues
        if 'nl_object_variation' in transformation_type:
            consistent_problem = self._fix_entity_consistency(
                original_problem, cleaned_problem, original_component
            )
        else:
            consistent_problem = cleaned_problem
        
        return consistent_problem
    
    def _fix_formatting_issues(self, problem: str) -> str:
        """Fix obvious formatting problems like redundant AM/PM"""
        fixed_problem = problem
        
        # Fix time format redundancies
        for pattern, replacement in self.time_conflict_patterns:
            fixed_problem = re.sub(pattern, replacement, fixed_problem, flags=re.IGNORECASE)
        
        # Fix double periods, spaces, etc.
        fixed_problem = re.sub(r'\.\.+', '.', fixed_problem)
        fixed_problem = re.sub(r'\s+', ' ', fixed_problem)
        fixed_problem = fixed_problem.strip()
        
        return fixed_problem
    
    def _fix_entity_consistency(self, original: str, modified: str, original_component: str) -> str:
        """Fix inconsistent entity transformations using semantic analysis"""
        
        # Find what the original component was changed to
        attempted_replacement = self._find_attempted_replacement(original, modified, original_component)
        
        if not attempted_replacement or attempted_replacement == original_component:
            return modified
        
        # Find all semantically related entities that should also change
        related_entities = self._find_related_entities(original, original_component)
        
        # Apply consistent transformations
        consistent_problem = self._apply_consistent_entity_transformation(
            original, modified, original_component, attempted_replacement, related_entities
        )
        
        return consistent_problem
    
    def _find_attempted_replacement(self, original: str, modified: str, original_component: str) -> str:
        """Find what the original component was changed to"""
        
        # Remove markup like ** from components
        clean_original = original_component.replace('**', '').strip()
        
        # Simple approach: find words that are in modified but not original
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        modified_words = set(re.findall(r'\b\w+\b', modified.lower()))
        
        # Find words that were added
        new_words = modified_words - original_words
        
        # Find words that were removed (should include our target)
        removed_words = original_words - modified_words
        
        if clean_original.lower() in removed_words and len(new_words) >= 1:
            # Return the most likely replacement (longest new word)
            return max(new_words, key=len) if new_words else clean_original
        
        return clean_original
    
    def _find_related_entities(self, problem: str, original_component: str) -> list:
        """Find entities related to the original component using advanced NLP libraries"""
        
        related_entities = []
        clean_component = original_component.replace('**', '').strip()
        
        # Method 1: spaCy-based semantic and dependency analysis
        if self.nlp and self.semantic_detectors.get('spacy_deps'):
            related_entities.extend(self._find_related_via_spacy(problem, clean_component))
        
        # Method 2: WordNet-based semantic similarity  
        if self.semantic_detectors.get('wordnet'):
            related_entities.extend(self._find_related_via_wordnet(problem, clean_component))
        
        # Method 3: SymPy-based mathematical relationships
        if self.semantic_detectors.get('sympy'):
            related_entities.extend(self._find_related_via_sympy(problem, clean_component))
        
        # Method 4: Fallback pattern-based detection for count-unit relationships
        related_entities.extend(self._find_related_via_patterns(problem, clean_component))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity, pos in related_entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append((entity, pos))
        
        return unique_entities
    
    def _find_related_via_spacy(self, problem: str, component: str) -> list:
        """Find related entities using spaCy's dependency parsing and NER"""
        related = []
        
        try:
            doc = self.nlp(problem)
            component_tokens = self.nlp(component)
            
            # Find the component in the document
            component_spans = []
            for i, token in enumerate(doc):
                if token.text.lower() == component.split()[0].lower():
                    # Check if this is the start of our component
                    component_text = doc[i:i+len(component.split())].text
                    if component.lower() in component_text.lower():
                        component_spans.append(doc[i:i+len(component.split())])
            
            for span in component_spans:
                # Find entities with similar dependency roles
                for token in span:
                    # Look for tokens with similar dependencies and POS tags
                    for other_token in doc:
                        if (other_token.dep_ == token.dep_ and 
                            other_token.pos_ == token.pos_ and 
                            other_token.text != token.text and
                            other_token.head.pos_ == token.head.pos_):
                            
                            # Check if it's part of a count-unit structure
                            if self._is_similar_unit_structure(token, other_token):
                                related.append((other_token.text, (other_token.idx, other_token.idx + len(other_token.text))))
                
                # Also look for compound and coordination relationships
                for token in span:
                    for child in token.children:
                        if child.dep_ in ['compound', 'conj', 'appos']:
                            related.append((child.text, (child.idx, child.idx + len(child.text))))
                    
                    # Look at parent relationships
                    if token.head.dep_ in ['compound', 'conj', 'appos']:
                        related.append((token.head.text, (token.head.idx, token.head.idx + len(token.head.text))))
        
        except Exception as e:
            logger.debug(f"    spaCy analysis error: {e}")
        
        return related
    
    def _find_related_via_wordnet(self, problem: str, component: str) -> list:
        """Find semantically related entities using WordNet"""
        related = []
        
        try:
            wordnet = self.semantic_detectors['wordnet']
            
            # Extract the main noun from the component
            component_words = component.lower().split()
            main_word = None
            
            # Find the main noun (skip articles, numbers, adjectives)
            for word in reversed(component_words):  # Start from the end (usually the main noun)
                if not re.match(r'^(a|an|the|\d+)$', word):
                    main_word = word
                    break
            
            if not main_word:
                return related
            
            # Get WordNet synsets for the main word
            synsets = wordnet.synsets(main_word, pos=wordnet.NOUN)
            
            if not synsets:
                return related
            
            # Find words in the problem that are similar to our component
            doc = self.nlp(problem) if self.nlp else None
            problem_words = re.findall(r'\b\w+\b', problem.lower())
            
            for word in problem_words:
                if word == main_word:
                    continue
                
                # Check semantic similarity using WordNet
                word_synsets = wordnet.synsets(word, pos=wordnet.NOUN)
                for syn1 in synsets[:2]:  # Check top 2 synsets
                    for syn2 in word_synsets[:2]:
                        similarity = syn1.wup_similarity(syn2)
                        if similarity and similarity > 0.6:  # High similarity threshold
                            # Find this word in context
                            for match in re.finditer(rf'\b{re.escape(word)}\b', problem, re.IGNORECASE):
                                related.append((match.group(), (match.start(), match.end())))
                            break
        
        except Exception as e:
            logger.debug(f"    WordNet analysis error: {e}")
        
        return related
    
    def _find_related_via_sympy(self, problem: str, component: str) -> list:
        """Find mathematically related expressions using SymPy"""
        related = []
        
        try:
            # Look for mathematical relationships like unit conversions
            if re.search(r'\d+', component):
                # Find other numerical expressions in the problem
                number_patterns = [
                    r'\b\d+\s+\w+s?\b',  # "4 walls", "2 hours"
                    r'\b\d+\.\d+\s+\w+s?\b',  # "2.5 hours"
                ]
                
                for pattern in number_patterns:
                    matches = re.finditer(pattern, problem, re.IGNORECASE)
                    for match in matches:
                        expr = match.group()
                        if expr.lower() != component.lower():
                            # Check if they involve the same unit type
                            component_unit = re.search(r'\b(\w+)s?$', component)
                            expr_unit = re.search(r'\b(\w+)s?$', expr)
                            
                            if (component_unit and expr_unit and 
                                self._are_related_units(component_unit.group(1), expr_unit.group(1))):
                                related.append((expr, (match.start(), match.end())))
        
        except Exception as e:
            logger.debug(f"    SymPy analysis error: {e}")
        
        return related
    
    def _find_related_via_patterns(self, problem: str, component: str) -> list:
        """Fallback pattern-based detection for count-unit relationships"""
        related = []
        clean_component = component.lower()
        
        # Extract the unit from the component (remove numbers, articles, etc.)
        unit = re.sub(r'^(a|an|the|\d+)\s+', '', clean_component.strip())
        
        # Find all count-unit patterns with this unit (both singular and plural)
        base_unit = unit.rstrip('s') if unit.endswith('s') else unit
        
        patterns = [
            rf'\b\d+\s+{re.escape(base_unit)}s?\b',
            rf'\b(a|an|one)\s+{re.escape(base_unit)}\b',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, problem, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0)
                # Skip if this is the same as the original component (already changed)
                if clean_component not in entity_text.lower():
                    related.append((entity_text, (match.start(), match.end())))
        
        return related
    
    def _is_similar_unit_structure(self, token1, token2) -> bool:
        """Check if two tokens have similar unit structure using spaCy"""
        
        # Check if both are preceded by numbers
        def has_number_before(token):
            for left_token in token.lefts:
                if left_token.pos_ == 'NUM' or left_token.like_num:
                    return True
            return False
        
        def has_article_before(token):
            for left_token in token.lefts:
                if left_token.pos_ == 'DET':
                    return True
            return False
        
        # Similar structure if both have numbers or both have articles
        return ((has_number_before(token1) and has_number_before(token2)) or
                (has_article_before(token1) and has_article_before(token2)))
    
    def _are_related_units(self, unit1: str, unit2: str) -> bool:
        """Check if two units are semantically related"""
        
        # Check if they're the same unit (singular/plural)
        base1 = unit1.rstrip('s')
        base2 = unit2.rstrip('s')
        
        if base1 == base2:
            return True
        
        # Use WordNet to check semantic similarity if available
        if self.semantic_detectors.get('wordnet'):
            try:
                wordnet = self.semantic_detectors['wordnet']
                synsets1 = wordnet.synsets(unit1, pos=wordnet.NOUN)
                synsets2 = wordnet.synsets(unit2, pos=wordnet.NOUN)
                
                for syn1 in synsets1[:2]:
                    for syn2 in synsets2[:2]:
                        similarity = syn1.wup_similarity(syn2)
                        if similarity and similarity > 0.7:
                            return True
            except:
                pass
        
        return False
    
    def _apply_consistent_entity_transformation(self, original: str, modified: str, 
                                              original_component: str, replacement: str, 
                                              related_entities: list) -> str:
        """Apply consistent transformation to all related entities"""
        
        if not related_entities:
            return modified
        
        result = modified
        
        # For each related entity, apply a consistent transformation
        for entity_text, position in related_entities:
            transformed_entity = self._transform_related_entity(entity_text, original_component, replacement)
            
            if transformed_entity and transformed_entity != entity_text:
                # Replace in the result text
                result = result.replace(entity_text, transformed_entity, 1)
        
        return result
    
    def _transform_related_entity(self, entity: str, original_component: str, replacement: str) -> str:
        """Transform a related entity consistently with the main transformation"""
        
        # Extract patterns from the original transformation
        original_clean = original_component.replace('**', '').strip()
        
        # Handle singular/plural relationships
        if original_clean in entity.lower():
            # Direct substitution
            return re.sub(re.escape(original_clean), replacement, entity, flags=re.IGNORECASE)
        
        # Handle singular/plural relationships
        if original_clean.endswith('s') and not entity.lower().endswith('s'):
            # Original was plural, entity is singular
            singular_replacement = replacement.rstrip('s') if replacement.endswith('s') else replacement
            base_original = original_clean.rstrip('s')
            return re.sub(re.escape(base_original), singular_replacement, entity, flags=re.IGNORECASE)
        
        elif not original_clean.endswith('s') and entity.lower().endswith('s'):
            # Original was singular, entity is plural
            plural_replacement = replacement + 's' if not replacement.endswith('s') else replacement
            base_original = original_clean
            return re.sub(re.escape(base_original), replacement, entity, flags=re.IGNORECASE)
        
        return entity

# Import model clients - lazy loading to avoid GPU conflicts
MODEL_CLIENT_AVAILABLE = None  # Will be checked when actually needed

def _check_model_client_availability():
    """Check if model clients are available - only called when actually needed."""
    global MODEL_CLIENT_AVAILABLE
    if MODEL_CLIENT_AVAILABLE is None:
        try:
            from benchdrift.models.model_client import VLLMClient, ModelClientFactory, RITSClient
            from benchdrift.models.model_config_manager import ModelConfigManager
            MODEL_CLIENT_AVAILABLE = True
        except ImportError as e:
            logger.debug(f"Warning: Model clients not available: {e}")
            MODEL_CLIENT_AVAILABLE = False
    return MODEL_CLIENT_AVAILABLE

def create_model_client_for_variations(client_type: str = None, model_name: str = None, max_model_len: int = 8192):
    """Create model client for variations using synthesis detector's setup"""
    if not _check_model_client_availability():
        logger.debug("Model clients not available - using deterministic variations only")
        return None

    try:
        # Import model clients locally to ensure they are available
        from benchdrift.models.model_client import VLLMClient, ModelClientFactory, RITSClient
        from benchdrift.models.model_config_manager import ModelConfigManager

        # Initialize model config manager
        model_config = ModelConfigManager()

        # Get recommended model and client for variation generation
        recommended_model, recommended_client = model_config.get_recommended_model_client(
            preferred_model=model_name,
            preferred_client=client_type,
            task='variation_generation'
        )

        logger.debug(f"🎯 Creating model client: {recommended_model} with {recommended_client}")

        # Get client settings
        client_settings = model_config.get_client_settings(recommended_client)

        if recommended_client == 'rits':
            max_workers = client_settings.get('max_workers', 5)
            max_new_tokens = client_settings.get('max_new_tokens', 1000)
            return RITSClient(recommended_model, max_workers=max_workers, max_new_tokens=max_new_tokens)
        elif recommended_client == 'vllm':
            try:
                return ModelClientFactory.create_client('vllm', recommended_model, max_model_len=max_model_len)
            except:
                return VLLMClient(recommended_model, max_model_len=max_model_len)
        else:
            # Try factory for other clients
            try:
                return ModelClientFactory.create_client(recommended_client, recommended_model)
            except Exception as e:
                logger.debug(f"Factory creation failed: {e}")
                return None

    except Exception as e:
        logger.debug(f"Error creating model client: {e}")
        return None

class TemporalDomainVariationEngine:
    """
    Specialized variation engine for temporal problems.
    Handles time, date, and duration pattern detection and transformations.
    """
    
    def __init__(self):
        self.ureg = UnitRegistry()
        
        # Load spaCy model for dependency parsing
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            logger.debug("✅ spaCy loaded for dependency parsing")
        except:
            self.nlp = None  
            self.spacy_available = False
            logger.debug("⚠️ spaCy not available - using basic transformations")
        
        # Comprehensive temporal pattern detection
        self.temporal_patterns = {
            'times_12hr': [
                r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)\b',  # 10:30 AM, 2:15 PM
                r'\b\d{1,2}:\d{2}:\d{2}\s*(AM|PM|am|pm)\b'  # 10:30:45 AM
            ],
            'times_24hr': [
                r'\b\d{1,2}:\d{2}(?::\d{2})?\b',  # 14:30, 14:30:45
                r'\b\d{2}:\d{2}\b'  # 08:00, 23:45
            ],
            'durations': [
                r'\b\d+\s+(hour|hours|minute|minutes|second|seconds|day|days|week|weeks|month|months|year|years)\b',
                r'\b\d+\s*hr?\b',  # 2h, 3hr
                r'\b\d+\s*min\b',  # 30min
                r'\b\d+\s*sec\b'   # 45sec
            ],
            'dates': [
                r'\b\d{4}-\d{2}-\d{2}\b',  # 2023-01-15
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # 1/15/2023
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ],
            'ordinals': [
                r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
                r'\b(\d+)(st|nd|rd|th)\b'  # 1st, 2nd, 3rd, 4th
            ],
            'schedule_words': [
                r'\b(daily|weekly|monthly|yearly|hourly|quarterly)\b',
                r'\bevery\s+\d+\s+(day|days|week|weeks|month|months|year|years|hour|hours)\b'
            ],
            'temporal_prep': [
                r'\b(before|after|during|until|since|while|prior\s+to|following|throughout|up\s+to|from|as)\b'
            ],
            'relative_time': [
                r'\b(yesterday|tomorrow|last\s+week|next\s+week|last\s+month|next\s+month|last\s+year|next\s+year)\b',
                r'\b\d+\s+days?\s+ago\b',
                r'\bin\s+\d+\s+days?\b'
            ],
            'event_frequency': [
                r'\bevery\s+\d+\s+months\b',
                r'\b(bi-weekly|bi-annually|every\s+other\s+day)\b'
            ]
        }
        
        # Comprehensive temporal transformations
        self.temporal_transformations = {
            'time_format_12_to_24': {
                '12:00 AM': '00:00', '1:00 AM': '01:00', '2:00 AM': '02:00',
                '10:00 AM': '10:00', '12:00 PM': '12:00', '1:00 PM': '13:00',
                '2:00 PM': '14:00', '10:00 PM': '22:00', '11:00 PM': '23:00',
                '6:00 AM': '06:00', '7:30 PM': '19:30', '8:15 AM': '08:15',
                '9:45 PM': '21:45', '11:30 AM': '11:30', '3:15 PM': '15:15'
            },
            'time_format_24_to_12': {
                '00:00': '12:00 AM', '01:00': '1:00 AM', '02:00': '2:00 AM',
                '10:00': '10:00 AM', '12:00': '12:00 PM', '13:00': '1:00 PM',
                '14:00': '2:00 PM', '22:00': '10:00 PM', '23:00': '11:00 PM',
                '06:00': '6:00 AM', '19:30': '7:30 PM', '08:15': '8:15 AM',
                '21:45': '9:45 PM', '11:30': '11:30 AM', '15:15': '3:15 PM'
            },
            'duration_units': {
                '1 hour': '60 minutes', '2 hours': '120 minutes', '3 hours': '180 minutes',
                '30 minutes': '1800 seconds', '45 minutes': '2700 seconds', '90 minutes': '1.5 hours',
                '1 day': '24 hours', '2 days': '48 hours', '1 week': '7 days',
                '2 weeks': '14 days', '1 month': '30 days', '3 months': '90 days',
                '6 months': '180 days', '1 year': '365 days', '2 years': '730 days'
            },
            'date_format_variations': {
                '2023-01-15': 'January 15, 2023', '2023-03-20': 'March 20, 2023',
                '2023-12-25': 'December 25, 2023', '2024-02-28': 'February 28, 2024',
                'January 1, 2023': '2023-01-01', 'March 15, 2024': '2024-03-15',
                'December 31, 2023': '2023-12-31', 'July 4, 2024': '2024-07-04'
            },
            'date_format_slash': {
                '2023-01-15': '01/15/2023', '2023-12-25': '12/25/2023',
                '2024-02-28': '02/28/2024', '2024-07-04': '07/04/2024',
                '01/15/2023': '2023-01-15', '12/25/2023': '2023-12-25',
                '02/28/2024': '2024-02-28', '07/04/2024': '2024-07-04'
            },
            'ordinal_sequence': {
                'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
                'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
                '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
                '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth'
            },
            'schedule_frequency': {
                'daily': 'every day', 'weekly': 'every 7 days', 'monthly': 'every 30 days',
                'yearly': 'every 365 days', 'hourly': 'every hour', 'quarterly': 'every 3 months',
                'every day': 'daily', 'every 7 days': 'weekly', 'every 30 days': 'monthly',
                'every 365 days': 'yearly', 'every hour': 'hourly', 'every 3 months': 'quarterly'
            },
            'temporal_prepositions': {
                'before': 'prior to', 'after': 'following', 'during': 'throughout',
                'until': 'up to', 'since': 'from', 'while': 'as',
                'prior to': 'before', 'following': 'after', 'throughout': 'during',
                'up to': 'until', 'from': 'since', 'as': 'while'
            },
            'relative_time': {
                'yesterday': '1 day ago', 'tomorrow': 'in 1 day', 'last week': '7 days ago',
                'next week': 'in 7 days', 'last month': '30 days ago', 'next month': 'in 30 days',
                'last year': '365 days ago', 'next year': 'in 365 days',
                '1 day ago': 'yesterday', 'in 1 day': 'tomorrow', '7 days ago': 'last week',
                'in 7 days': 'next week', '30 days ago': 'last month', 'in 30 days': 'next month'
            },
            'event_intervals': {
                'every 3 months': 'quarterly', 'every 6 months': 'bi-annually', 
                'every 2 weeks': 'bi-weekly', 'every 2 days': 'every other day',
                'quarterly': 'every 3 months', 'bi-annually': 'every 6 months',
                'bi-weekly': 'every 2 weeks', 'every other day': 'every 2 days'
            }
        }
    
    def detect_temporal_topics(self, problem: str) -> Dict[str, List[Tuple[str, int]]]:
        """Detect temporal patterns in the problem"""
        detected_topics = defaultdict(list)
        
        for topic, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem, re.IGNORECASE)
                for match in matches:
                    detected_topics[topic].append((match.group(), match.start()))
        
        return dict(detected_topics)
    
    def get_connected_time_components(self, problem: str, target_match: str, target_pos: int) -> List[Tuple[str, int]]:
        """Find time components connected by conjunctions using dependency parsing"""
        if not self.spacy_available:
            return [(target_match, target_pos)]
        
        try:
            doc = self.nlp(problem)
            connected_components = [(target_match, target_pos)]
            
            # Find the token corresponding to our target match
            target_tokens = []
            for token in doc:
                if target_pos <= token.idx < target_pos + len(target_match):
                    target_tokens.append(token)
            
            if not target_tokens:
                return connected_components
            
            # For each target token, find connected time components via conjunctions
            for target_token in target_tokens:
                # Check conjunction relationships (both as head and child)
                for token in doc:
                    if (token.dep_ == "conj" and token.head == target_token) or \
                       (target_token.dep_ == "conj" and target_token.head == token):
                        
                        # Check if this connected token is also a time component
                        token_with_num = self._get_time_component_with_number(doc, token)
                        if token_with_num and token_with_num not in [comp[0] for comp in connected_components]:
                            connected_components.append((token_with_num, token.idx))
            
            return connected_components
            
        except Exception as e:
            # Fallback to single component if parsing fails
            return [(target_match, target_pos)]
    
    def _get_time_component_with_number(self, doc, time_token) -> str:
        """Extract number + time unit (e.g., '30 minutes') from a time token"""
        try:
            # Look for numeric modifiers (nummod) of this token
            for child in time_token.children:
                if child.dep_ == "nummod":
                    # Return number + time unit
                    return f"{child.text} {time_token.text}"
            
            # If no nummod found, check if token itself is part of time expression
            for token in doc:
                if token.dep_ == "nummod" and token.head == time_token:
                    return f"{token.text} {time_token.text}"
                    
            return None
        except:
            return None
    
    def find_composite_time_expression(self, problem: str, match_text: str, match_pos: int) -> str:
        """Find the complete composite time expression that contains this component"""
        # First try pattern-based detection (most reliable for temporal expressions)
        composite_patterns = [
            r'\d+\s+hours?,\s*\d+\s+minutes?,\s*(and\s+)?\d+\s+seconds?',  # "2 hours, 30 minutes, and 45 seconds"
            r'\d+\s+minutes?,\s*(and\s+)?\d+\s+seconds?',  # "30 minutes and 45 seconds"
            r'\d+\s+hours?,\s*(and\s+)?\d+\s+minutes?',   # "2 hours and 30 minutes"
        ]
        
        for pattern in composite_patterns:
            matches = list(re.finditer(pattern, problem, re.IGNORECASE))
            for composite_match in matches:
                # Check if our individual match is within this composite
                if (composite_match.start() <= match_pos < composite_match.end()):
                    return composite_match.group().strip()
        
        # Fallback to dependency-based detection
        dependency_composite = self.find_composite_temporal_expression_via_dependencies(problem, match_text, match_pos)
        if dependency_composite:
            return dependency_composite
        
        return None
    
    def find_composite_temporal_expression_via_dependencies(self, problem: str, match_text: str, match_pos: int) -> str:
        """Use spaCy dependency parsing to find composite temporal expressions containing this component"""
        if not self.spacy_available:
            return None
        
        try:
            doc = self.nlp(problem)
            
            # Find the token(s) corresponding to our match
            target_tokens = []
            for token in doc:
                if match_pos <= token.idx < match_pos + len(match_text):
                    target_tokens.append(token)
            
            if not target_tokens:
                return None
            
            # For each target token, find the complete temporal expression it belongs to
            for target_token in target_tokens:
                # Get the root of the expression containing this token
                temporal_expr_tokens = self._get_connected_temporal_expression_tokens(target_token)
                
                if len(temporal_expr_tokens) > 1:  # More than just the target token
                    # Sort by position and reconstruct the expression
                    temporal_expr_tokens_list = sorted(temporal_expr_tokens, key=lambda t: t.idx)
                    start_idx = temporal_expr_tokens_list[0].idx
                    end_idx = temporal_expr_tokens_list[-1].idx + len(temporal_expr_tokens_list[-1].text)
                    composite_expr = problem[start_idx:end_idx].strip()
                    
                    if composite_expr != match_text:  # It's actually composite
                        return composite_expr
            
            return None
            
        except Exception as e:
            logger.debug(f"    Exception in find_composite_temporal_expression_via_dependencies: {e}")
            return None
    
    def _get_connected_temporal_expression_tokens(self, token):
        """Get all tokens that are part of the same temporal expression using spaCy dependencies"""
        expression_tokens = {token}
        visited = {token}
        to_process = [token]
        
        while to_process:
            current_token = to_process.pop(0)
            
            # Add children that are part of temporal expressions (using TEMPORAL_COMPOSITE_RULES)
            for child in current_token.children:
                if child not in visited:
                    # Include temporal coordinations, conjunctions, compounds, numbers with time units
                    all_temporal_lemmas = (
                        TEMPORAL_COMPOSITE_RULES['time_units'] +
                        TEMPORAL_COMPOSITE_RULES['time_units_abbrev'] +
                        TEMPORAL_COMPOSITE_RULES['eras'] +  # NEW: BC/AD from TOT data
                        TEMPORAL_COMPOSITE_RULES['seasons'] +
                        TEMPORAL_COMPOSITE_RULES['named_times'] +
                        TEMPORAL_COMPOSITE_RULES['relative_time'] +
                        TEMPORAL_COMPOSITE_RULES['time_phrases'] +
                        TEMPORAL_COMPOSITE_RULES['larger_units']
                    )
                    if (child.dep_ in TEMPORAL_COMPOSITE_RULES['dependencies'] or
                        child.pos_ in TEMPORAL_COMPOSITE_RULES['pos_tags'] or
                        child.pos_ == 'CCONJ' or
                        child.text in [',', 'and', 'or'] + TEMPORAL_COMPOSITE_RULES['separators'] or
                        child.lemma_ in all_temporal_lemmas or
                        child.text in TEMPORAL_COMPOSITE_RULES['eras']):  # Catch BC/AD without lemmatization
                        expression_tokens.add(child)
                        visited.add(child)
                        to_process.append(child)
            
            # For temporal expressions, look for coordinated time units (using rules)
            all_time_units = TEMPORAL_COMPOSITE_RULES['time_units'] + TEMPORAL_COMPOSITE_RULES['larger_units']
            if current_token.lemma_ in all_time_units or current_token.pos_ == 'NUM':
                for sibling in current_token.head.children:
                    if (sibling not in visited and sibling != current_token):
                        # Include other temporal units and numbers as siblings (using rules)
                        if (sibling.pos_ in ['NUM'] or
                            sibling.lemma_ in all_time_units or
                            sibling.text in [',', 'and', 'or'] + TEMPORAL_COMPOSITE_RULES['separators'] or
                            sibling.text in TEMPORAL_COMPOSITE_RULES['eras'] or  # BC/AD
                            (sibling.dep_ in ['nummod', 'conj', 'appos'] and sibling.pos_ in ['NUM', 'NOUN'])):
                            expression_tokens.add(sibling)
                            visited.add(sibling)
                            to_process.append(sibling)

        # Filter to keep only tokens that are likely part of temporal expressions (using rules)
        all_temporal_lemmas = (
            TEMPORAL_COMPOSITE_RULES['time_units'] +
            TEMPORAL_COMPOSITE_RULES['time_units_abbrev'] +
            TEMPORAL_COMPOSITE_RULES['larger_units'] +
            TEMPORAL_COMPOSITE_RULES['named_times'] +
            TEMPORAL_COMPOSITE_RULES['time_phrases']
        )
        all_temporal_texts = (
            TEMPORAL_COMPOSITE_RULES['eras'] +
            TEMPORAL_COMPOSITE_RULES['separators'] +
            [',', 'and', 'or']
        )

        temporal_expression_tokens = set()
        for tok in expression_tokens:
            if (tok.pos_ in TEMPORAL_COMPOSITE_RULES['pos_tags'] or
                tok.text in all_temporal_texts or
                tok.lemma_ in all_temporal_lemmas or
                tok.dep_ in TEMPORAL_COMPOSITE_RULES['dependencies']):
                temporal_expression_tokens.add(tok)
        
        # If we have more than just the original token, return the temporal subset
        if len(temporal_expression_tokens) > 1:
            return temporal_expression_tokens
        else:
            return {token}  # Just return the original token if no temporal expression found
    
    def should_skip_individual_transformation(self, problem: str, match_text: str, match_pos: int) -> bool:
        """Check if this component is part of a composite time expression and should be skipped"""
        composite = self.find_composite_time_expression(problem, match_text, match_pos)
        if composite:
            logger.debug(f"    📋 Found composite: '{composite}' - will handle with model")
            return True
        return False
    
    def apply_temporal_transformations(self, problem: str, detected_topics: Dict[str, List[Tuple[str, int]]]) -> List[Dict[str, Any]]:
        """Apply deterministic temporal transformations"""
        variations = []
        
        for topic, matches in detected_topics.items():
            for match_text, match_pos in matches:
                
                # Time format conversions
                if topic == 'times_12hr' and match_text in self.temporal_transformations['time_format_12_to_24']:
                    new_text = problem.replace(match_text, self.temporal_transformations['time_format_12_to_24'][match_text], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'time_12hr_to_24hr',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['time_format_12_to_24'][match_text]
                    })
                
                elif topic == 'times_24hr' and match_text in self.temporal_transformations['time_format_24_to_12']:
                    new_text = problem.replace(match_text, self.temporal_transformations['time_format_24_to_12'][match_text], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'time_24hr_to_12hr',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['time_format_24_to_12'][match_text]
                    })
                
                # Duration unit conversions with context awareness
                elif topic == 'durations':
                    # CONTEXT AWARENESS: Skip if part of composite time expression
                    if self.should_skip_individual_transformation(problem, match_text, match_pos):
                        logger.debug(f"    🔗 Skipping '{match_text}' - part of composite time expression")
                        continue
                    
                    # NEW PARSER-BASED APPROACH with context awareness
                    try:
                        # Try to parse duration with pint
                        quantity = self.ureg.parse_expression(match_text)
                        
                        # Define conversion targets based on original unit
                        if 'hour' in str(quantity.units):
                            conversions = [('minute', 'minutes'), ('second', 'seconds')]
                        elif 'minute' in str(quantity.units):
                            conversions = [('hour', 'hours'), ('second', 'seconds')]
                        elif 'day' in str(quantity.units):
                            conversions = [('hour', 'hours'), ('week', 'weeks')]
                        elif 'week' in str(quantity.units):
                            conversions = [('day', 'days'), ('month', 'months')]
                        else:
                            conversions = []
                        
                        for target_unit, unit_name in conversions:
                            try:
                                converted = quantity.to(target_unit)
                                # Format nicely (e.g., "120.0 minute" -> "120 minutes")
                                value = converted.magnitude
                                if value == int(value):
                                    value = int(value)
                                new_duration = f"{value} {unit_name if value != 1 else unit_name[:-1]}"
                                
                                if new_duration != match_text:
                                    new_text = problem.replace(match_text, new_duration, 1)
                                    variations.append({
                                        'modified_problem': new_text,
                                        'transformation_type': f'duration_to_{target_unit}',
                                        'original_component': match_text,
                                        'new_component': new_duration
                                    })
                            except Exception:
                                continue
                                
                    except Exception:
                        # Fallback to rigid matching if parsing fails (also with context check)
                        if match_text in self.temporal_transformations.get('duration_units', {}):
                            new_text = problem.replace(match_text, self.temporal_transformations['duration_units'][match_text], 1)
                            variations.append({
                                'modified_problem': new_text,
                                'transformation_type': 'duration_unit_fallback',
                                'original_component': match_text,
                                'new_component': self.temporal_transformations['duration_units'][match_text]
                            })
                
                # Date format variations
                elif topic == 'dates':
                    # OLD RIGID APPROACH (commented out)
                    # if match_text in self.temporal_transformations['date_format_variations']:
                    #     new_text = problem.replace(match_text, self.temporal_transformations['date_format_variations'][match_text], 1)
                    #     variations.append({
                    #         'modified_problem': new_text,
                    #         'transformation_type': 'date_format_variation',
                    #         'original_component': match_text,
                    #         'new_component': self.temporal_transformations['date_format_variations'][match_text]
                    #     })
                    # if match_text in self.temporal_transformations['date_format_slash']:
                    #     new_text = problem.replace(match_text, self.temporal_transformations['date_format_slash'][match_text], 1)
                    #     variations.append({
                    #         'modified_problem': new_text,
                    #         'transformation_type': 'date_format_slash',
                    #         'original_component': match_text,
                    #         'new_component': self.temporal_transformations['date_format_slash'][match_text]
                    #     })
                    
                    # NEW PARSER-BASED APPROACH
                    try:
                        parsed_date = date_parser.parse(match_text)
                        # Generate multiple format variations
                        formats = [
                            ("%Y-%m-%d", "iso_format"),           # 2023-01-15
                            ("%m/%d/%Y", "us_slash_format"),     # 01/15/2023
                            ("%B %d, %Y", "written_format"),    # January 15, 2023
                            ("%d-%b-%Y", "day_month_year")       # 15-Jan-2023
                        ]
                        
                        for format_str, format_name in formats:
                            new_date = parsed_date.strftime(format_str)
                            if new_date != match_text:  # Only add if different
                                new_text = problem.replace(match_text, new_date, 1)
                                variations.append({
                                    'modified_problem': new_text,
                                    'transformation_type': f'date_{format_name}',
                                    'original_component': match_text,
                                    'new_component': new_date
                                })
                    except (ValueError, TypeError):
                        # Fallback to rigid matching if parsing fails
                        if match_text in self.temporal_transformations.get('date_format_variations', {}):
                            new_text = problem.replace(match_text, self.temporal_transformations['date_format_variations'][match_text], 1)
                            variations.append({
                                'modified_problem': new_text,
                                'transformation_type': 'date_format_fallback',
                                'original_component': match_text,
                                'new_component': self.temporal_transformations['date_format_variations'][match_text]
                            })
                
                # Ordinal sequence transformations
                elif topic == 'ordinals' and match_text.lower() in self.temporal_transformations['ordinal_sequence']:
                    new_text = problem.replace(match_text, self.temporal_transformations['ordinal_sequence'][match_text.lower()], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'ordinal_sequence',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['ordinal_sequence'][match_text.lower()]
                    })
                
                # Schedule frequency transformations
                elif topic == 'schedule_words' and match_text.lower() in self.temporal_transformations['schedule_frequency']:
                    new_text = problem.replace(match_text, self.temporal_transformations['schedule_frequency'][match_text.lower()], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'schedule_frequency',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['schedule_frequency'][match_text.lower()]
                    })
                
                # Temporal preposition transformations
                elif topic == 'temporal_prep' and match_text.lower() in self.temporal_transformations['temporal_prepositions']:
                    new_text = problem.replace(match_text, self.temporal_transformations['temporal_prepositions'][match_text.lower()], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'temporal_preposition',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['temporal_prepositions'][match_text.lower()]
                    })
                
                # Relative time transformations
                elif topic == 'relative_time' and match_text.lower() in self.temporal_transformations['relative_time']:
                    new_text = problem.replace(match_text, self.temporal_transformations['relative_time'][match_text.lower()], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'relative_time',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['relative_time'][match_text.lower()]
                    })
                
                # Event interval transformations
                elif topic == 'event_frequency' and match_text.lower() in self.temporal_transformations['event_intervals']:
                    new_text = problem.replace(match_text, self.temporal_transformations['event_intervals'][match_text.lower()], 1)
                    variations.append({
                        'modified_problem': new_text,
                        'transformation_type': 'event_interval',
                        'original_component': match_text,
                        'new_component': self.temporal_transformations['event_intervals'][match_text.lower()]
                    })
        
        return variations
    
    def collect_composite_time_expressions(self, problem: str, detected_topics: Dict[str, List[Tuple[str, int]]]) -> Set[str]:
        """Collect all composite time expressions found in the problem"""
        composites = set()
        
        if 'durations' in detected_topics:
            for match_text, match_pos in detected_topics['durations']:
                composite = self.find_composite_time_expression(problem, match_text, match_pos)
                if composite:
                    composites.add(composite)
        
        return composites

class MathDomainVariationEngine:
    """
    Specialized variation engine for mathematical problems.
    Handles pattern detection and equivalent transformations for math topics.
    Uses SymPy for mathematical expression parsing + spaCy dependency parsing as fallback.
    """
    
    def __init__(self):
        # Load SymPy for mathematical expression parsing
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr
            self.sympy = sp
            self.parse_expr = parse_expr
            self.sympy_available = True
            logger.debug("✅ SymPy loaded for mathematical expression parsing")
        except ImportError:
            self.sympy = None
            self.parse_expr = None
            self.sympy_available = False
            logger.debug("⚠️ SymPy not available - using spaCy dependency parsing only")
        
        # Load spaCy for dependency parsing as fallback
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except:
            self.nlp = None  
            self.spacy_available = False
        # Mathematical topic patterns for detection
        self.math_patterns = {
            'fractions': [
                r'\b\d+/\d+\b',  # 3/4, 1/2
                r'\\frac\{\d+\}\{\d+\}',  # LaTeX fractions
                r'\b(one|two|three|four|five|six|seven|eight|nine)-?(half|third|fourth|fifth|sixth|seventh|eighth|ninth)s?\b',
                r'\b(half|quarter|third)\b'
            ],
            'decimals': [
                r'\b\d+\.\d+\b',  # 3.14, 0.5
                r'\b0\.\d+\b'     # 0.25, 0.333
            ],
            'percentages': [
                r'\b\d+(\.\d+)?%\b',  # 25%, 33.3%
                r'\b\d+(\.\d+)?\s*percent\b'
            ],
            'number_words': [
                r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'
            ],
            'numbers': [
                r'\b\d+\b'  # Plain integers: 15, 25, 100, etc.
            ],
            'trigonometry': [
                r'\b(sin|cos|tan|cot|sec|csc)\s*\([^)]+\)',
                r'\b(sine|cosine|tangent|cotangent|secant|cosecant)\b',
                r'\b\d+\s*(degree|degrees|°|radian|radians)\b'
            ],
            'algebra': [
                r'\b[a-zA-Z]\s*[\+\-\*\/\^]\s*[a-zA-Z0-9]',  # x + 2, y^2
                r'\b[a-zA-Z]\s*\^\s*\d+\b',  # x^2, y^3 (only with exponents)
                r'\b(solve|find|determine)\s+(for\s+)?[a-zA-Z]\b'  # solve for x
            ],
            'geometry': [
                r'\b(circle|triangle|rectangle|square|polygon|sphere|cube|cylinder|cone)\b',
                r'\b(radius|diameter|circumference|area|perimeter|volume|surface area)\b',
                r'\b(angle|degrees|perpendicular|parallel|congruent|similar)\b'
            ],
            'statistics': [
                r'\b(mean|median|mode|average|standard deviation|variance)\b',
                r'\b(probability|likely|unlikely|chance|odds)\b',
                r'\b\d+(\.\d+)?%\s*(chance|probability|likely)\b'
            ],
            'calculus': [
                r'\b(derivative|integral|limit|maximum|minimum|optimize)\b',
                r'\b(rate of change|slope|tangent line)\b',
                r'\bd/dx\b|\bf\'|\\frac\{d\}\{dx\}'
            ]
        }
        
        # Mathematical equivalent transformations
        self.math_transformations = {
            'fraction_to_decimal': {
                '1/2': '0.5', '1/4': '0.25', '3/4': '0.75', '1/3': '0.333', 
                '2/3': '0.667', '1/5': '0.2', '2/5': '0.4', '3/5': '0.6', '4/5': '0.8',
                '1/8': '0.125', '3/8': '0.375', '5/8': '0.625', '7/8': '0.875'
            },
            'decimal_to_fraction': {
                '0.5': '1/2', '0.25': '1/4', '0.75': '3/4', '0.2': '1/5', 
                '0.4': '2/5', '0.6': '3/5', '0.8': '4/5', '0.125': '1/8',
                '0.333': '1/3', '0.667': '2/3'
            },
            'fraction_to_percentage': {
                '1/2': '50%', '1/4': '25%', '3/4': '75%', '1/5': '20%',
                '2/5': '40%', '3/5': '60%', '4/5': '80%', '1/3': '33.3%', '2/3': '66.7%'
            },
            'percentage_to_fraction': {
                '50%': '1/2', '25%': '1/4', '75%': '3/4', '20%': '1/5',
                '40%': '2/5', '60%': '3/5', '80%': '4/5', '33.3%': '1/3', '66.7%': '2/3'
            },
            'number_to_word': {
                '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
                '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
                '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
                '30': 'thirty', '40': 'forty', '50': 'fifty', '100': 'one hundred'
            },
            'word_to_number': {
                'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
                'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
                'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
                'thirty': '30', 'forty': '40', 'fifty': '50'
            },
            'trigonometric_equivalents': {
                'sin(30°)': 'sin(π/6)', 'sin(45°)': 'sin(π/4)', 'sin(60°)': 'sin(π/3)', 'sin(90°)': 'sin(π/2)',
                'cos(30°)': 'cos(π/6)', 'cos(45°)': 'cos(π/4)', 'cos(60°)': 'cos(π/3)', 'cos(90°)': 'cos(π/2)',
                'sin(π/6)': 'sin(30°)', 'sin(π/4)': 'sin(45°)', 'sin(π/3)': 'sin(60°)', 'sin(π/2)': 'sin(90°)'
            },
            'algebraic_equivalents': {
                'x^2': 'x·x', 'x·x': 'x²', 'y^3': 'y·y·y', 'x^(1/2)': '√x', '√x': 'x^(1/2)',
                'x + x': '2x', '2x': 'x + x', '3x': 'x + x + x'
            },
            'geometry_equivalents': {
                # SAFE NEAR-TRANSFER EQUIVALENTS (mathematically identical)
                'rectangle': 'rectangular shape',  # Safe: same properties
                'rectangular shape': 'rectangle',
                'circle': 'circular shape', 'circular shape': 'circle',
                'triangle': 'triangular shape', 'triangular shape': 'triangle', 
                'square': 'square shape', 'square shape': 'square',
                
                # DESCRIPTIVE NEAR-TRANSFER (same mathematical meaning)
                'area': 'total area', 'total area': 'area',  # Both 2D
                'perimeter': 'boundary length', 'boundary length': 'perimeter',
                'length': 'side length', 'side length': 'length',
                'width': 'side width', 'side width': 'width',
                'height': 'vertical dimension', 'vertical dimension': 'height',
                'radius': 'distance from center', 'distance from center': 'radius',
                
                # REMOVED UNSAFE MAPPINGS that caused errors:
                # 'rectangle': 'quadrilateral' - WRONG: rectangle ⊂ quadrilateral
                # 'area': 'surface area' - WRONG: 2D vs 3D concepts  
                # 'perimeter': 'circumference' - WRONG: polygon vs circle
            }
        }
    
    def detect_math_topics(self, problem: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Detect mathematical topics/patterns in the problem.
        Returns dict with topic types and their matches.
        Uses context-aware filtering to avoid false positives.
        """
        detected_topics = defaultdict(list)
        
        for topic, patterns in self.math_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem, re.IGNORECASE)
                for match in matches:
                    match_text = match.group()
                    match_pos = match.start()
                    
                    # Apply context-aware filtering to avoid false positives
                    if self._is_valid_mathematical_match(problem, match_text, match_pos, topic):
                        detected_topics[topic].append((match_text, match_pos))
        
        return dict(detected_topics)
    
    def _is_valid_mathematical_match(self, problem: str, match_text: str, match_pos: int, topic: str) -> bool:
        """
        Context-aware validation to filter out false positive mathematical matches.
        """
        # Get context around the match
        context_before = problem[max(0, match_pos - 20):match_pos].lower()
        context_after = problem[match_pos + len(match_text):match_pos + len(match_text) + 20].lower()
        match_lower = match_text.lower()
        
        # Special handling for algebra patterns
        if topic == 'algebra':
            # Single letters that are clearly not mathematical variables
            if len(match_text) == 1 and match_text.isalpha():
                # Check if it's at the beginning of a sentence (likely an article/determiner)
                if match_pos == 0 or problem[match_pos - 1] in '. ':
                    return False
                
                # Check if it's followed by common non-mathematical words
                non_math_words = ['project', 'person', 'team', 'group', 'company', 'student', 
                                'teacher', 'customer', 'item', 'product', 'service', 'task',
                                'problem', 'question', 'answer', 'result', 'solution',
                                'consistent', 'constant', 'regular', 'normal', 'standard']
                
                for word in non_math_words:
                    if context_after.startswith(' ' + word) or context_after.startswith(word):
                        return False
                
                # Check if it's part of common phrases
                common_phrases = ['a project', 'a person', 'a team', 'a group', 'a company',
                                'a student', 'a teacher', 'a customer', 'a consistent', 
                                'a regular', 'a normal', 'a standard']
                
                context_phrase = (context_before[-10:] + match_text + context_after[:10]).lower()
                for phrase in common_phrases:
                    if phrase in context_phrase:
                        return False
        
        # Special handling for fractions (avoid dates like 2023-12-31)
        if topic == 'fractions' and '/' in match_text:
            # Check if it looks like a date
            if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', problem[max(0, match_pos - 5):match_pos + len(match_text) + 5]):
                return False
        
        # Special handling for decimals (avoid version numbers, dates)
        if topic == 'decimals':
            # Avoid version numbers
            if 'version' in context_before or 'v' in context_before[-3:]:
                return False
            
            # Avoid dates with time
            if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', context_before + match_text + context_after):
                return False
        
        # Special handling for trigonometry - make sure it's not part of regular text
        if topic == 'trigonometry':
            # These are usually legitimate if they contain parentheses or degree symbols
            if '(' in match_text or '°' in match_text or 'sin' in match_lower or 'cos' in match_lower or 'tan' in match_lower:
                return True

        # Special handling for plain numbers - avoid matching parts of decimals, fractions, percentages
        if topic == 'numbers':
            # Check immediately before and after the match
            char_before = problem[match_pos - 1] if match_pos > 0 else ' '
            char_after = problem[match_pos + len(match_text)] if match_pos + len(match_text) < len(problem) else ' '

            # Skip if it's part of a decimal (e.g., "3" in "3.14")
            if char_before == '.' or char_after == '.':
                return False

            # Skip if it's part of a fraction (e.g., "3" in "3/4")
            if char_before == '/' or char_after == '/':
                return False

            # Skip if it's part of a percentage (e.g., "25" in "25%")
            if char_after == '%':
                return False

            # Skip if it's part of a time (e.g., "2" in "2:00")
            if char_before == ':' or char_after == ':':
                return False

        return True
    
    def apply_math_transformations(self, problem: str, detected_topics: Dict[str, List[Tuple[str, int]]]) -> List[Dict[str, Any]]:
        """
        Apply deterministic mathematical transformations that preserve answers.
        """
        variations = []
        
        for topic, matches in detected_topics.items():
            for match_text, match_pos in matches:
                
                # DEPENDENCY-BASED COMPOSITE DETECTION: Skip if part of composite expression
                if hasattr(self, 'should_skip_individual_transformation_via_deps') and self.should_skip_individual_transformation_via_deps(problem, match_text, match_pos):
                    continue
                
                # Apply relevant transformations based on detected topic
                if topic == 'fractions':
                    # Try fraction to decimal
                    if match_text in self.math_transformations['fraction_to_decimal']:
                        new_text = problem.replace(
                            match_text, 
                            self.math_transformations['fraction_to_decimal'][match_text], 
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'fraction_to_decimal',
                            'original_component': match_text,
                            'new_component': self.math_transformations['fraction_to_decimal'][match_text],
                            'topic': topic
                        })
                    
                    # Try fraction to percentage
                    if match_text in self.math_transformations['fraction_to_percentage']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['fraction_to_percentage'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'fraction_to_percentage',
                            'original_component': match_text,
                            'new_component': self.math_transformations['fraction_to_percentage'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'decimals':
                    # Try decimal to fraction
                    if match_text in self.math_transformations['decimal_to_fraction']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['decimal_to_fraction'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'decimal_to_fraction', 
                            'original_component': match_text,
                            'new_component': self.math_transformations['decimal_to_fraction'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'number_words':
                    # Try word to number
                    if match_text.lower() in self.math_transformations['word_to_number']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['word_to_number'][match_text.lower()],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'word_to_number',
                            'original_component': match_text,
                            'new_component': self.math_transformations['word_to_number'][match_text.lower()],
                            'topic': topic
                        })
                
                elif topic == 'trigonometry':
                    # Try trigonometric equivalents
                    if match_text in self.math_transformations['trigonometric_equivalents']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['trigonometric_equivalents'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'trigonometric_equivalent',
                            'original_component': match_text,
                            'new_component': self.math_transformations['trigonometric_equivalents'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'algebra':
                    # Try algebraic equivalents
                    if match_text in self.math_transformations['algebraic_equivalents']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['algebraic_equivalents'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'algebraic_equivalent',
                            'original_component': match_text,
                            'new_component': self.math_transformations['algebraic_equivalents'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'geometry':
                    # Try geometry term equivalents with validation
                    if match_text.lower() in self.math_transformations['geometry_equivalents']:
                        new_component = self.math_transformations['geometry_equivalents'][match_text.lower()]
                        
                        # Validate transformation before applying
                        if self._validate_geometry_transformation(match_text, new_component):
                            new_text = problem.replace(match_text, new_component, 1)
                            variations.append({
                                'modified_problem': new_text,
                                'transformation_type': 'geometry_equivalent',
                                'original_component': match_text,
                                'new_component': new_component,
                                'topic': topic
                            })
                        else:
                            logger.debug(f"    🚫 Skipped invalid geometry transformation: {match_text} → {new_component}")
        
        return variations
    
    def _validate_geometry_transformation(self, original_component: str, new_component: str) -> bool:
        """Validate geometric transformations to avoid unsafe mappings"""
        
        # Check for problematic geometric transformations
        geometry_issues = {
            ('rectangle', 'quadrilateral'): False,  # rectangle ⊂ quadrilateral (not equivalent)
            ('square', 'rectangle'): False,         # square ⊂ rectangle (not equivalent)
            ('area', 'surface area'): False,        # 2D vs 3D (not equivalent)
            ('perimeter', 'circumference'): False,  # polygon vs circle (not equivalent)
            ('triangle', 'polygon'): False,         # triangle ⊂ polygon (not equivalent)
        }
        
        # Check both directions
        key1 = (original_component.lower(), new_component.lower())
        key2 = (new_component.lower(), original_component.lower())
        
        if key1 in geometry_issues or key2 in geometry_issues:
            logger.debug(f"    ⚠️ Rejecting unsafe geometric transformation: '{original_component}' → '{new_component}'")
            return False
        
        # Check for completeness
        if len(new_component.strip()) < 2:
            logger.debug(f"    ⚠️ Rejecting incomplete transformation: '{original_component}' → '{new_component}'")
            return False
            
        return True
    
    def extract_mathematical_expressions_with_sympy(self, problem: str) -> List[Tuple[str, int, int]]:
        """
        Extract mathematical expressions from text using regex + SymPy validation.
        Returns list of (expression_text, start_pos, end_pos) tuples.
        """
        if not self.sympy_available:
            return []
        
        # Enhanced regex patterns to find potential mathematical expressions in mixed-domain text
        math_expression_patterns = [
            # Arithmetic expressions
            r'\b\d+(?:\.\d+)?(?:/\d+)?\s*[\+\-\*×÷]\s*\d+(?:\.\d+)?(?:/\d+)?\b(?:\s*[\+\-\*×÷]\s*\d+(?:\.\d+)?(?:/\d+)?\b)*',
            
            # Trigonometric expressions
            r'(?:sin|cos|tan|cot|sec|csc)\s*\([^)]+\)(?:\s*[\+\-\*×÷]\s*(?:sin|cos|tan|cot|sec|csc)\s*\([^)]+\))*',
            
            # Algebraic equations and expressions
            r'\b[a-zA-Z]\s*\^\s*\d+\s*[=]\s*\d+\b',  # x^2 = 16
            r'\b[a-zA-Z]\s*[\+\-\*]\s*[a-zA-Z0-9]+(?:\s*[\+\-\*]\s*[a-zA-Z0-9]+)*',  # x + 2, 2x + 3
            
            # Mathematical equivalences with mixed formats
            r'\d+(?:\.\d+)?%?\s+(?:equals?|=)\s+\d+/\d+(?:\s+(?:or|and|,)\s+\d+(?:\.\d+)?%?)*',
            r'\d+/\d+\s+(?:or|and|,)\s+\d+(?:\.\d+)?%?(?:\s+(?:or|and|,)\s+\d+(?:\.\d+)?%?)*',
            
            # Complex fractions and mixed operations
            r'\b\d+/\d+\s*[\+\-\*×÷]\s*\d+/\d+(?:\s*[\+\-\*×÷]\s*\d+/\d+)*\b',
        ]
        
        extracted_expressions = []
        
        for pattern in math_expression_patterns:
            for match in re.finditer(pattern, problem, re.IGNORECASE):
                expr_text = match.group().strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Validate with SymPy
                if self._validate_mathematical_expression_with_sympy(expr_text):
                    # Check for overlap with existing expressions (keep longer ones)
                    overlaps = False
                    for i, (existing_expr, existing_start, existing_end) in enumerate(extracted_expressions):
                        if not (end_pos <= existing_start or start_pos >= existing_end):  # Overlap detected
                            if len(expr_text) > len(existing_expr):
                                # Replace with longer expression
                                extracted_expressions[i] = (expr_text, start_pos, end_pos)
                            overlaps = True
                            break
                    
                    if not overlaps:
                        extracted_expressions.append((expr_text, start_pos, end_pos))
        
        # Sort by position
        extracted_expressions.sort(key=lambda x: x[1])
        return extracted_expressions
    
    def _validate_mathematical_expression_with_sympy(self, expr_text: str) -> bool:
        """Validate if the expression can be parsed by SymPy"""
        if not self.sympy_available:
            return False
        
        try:
            # Clean up expression for SymPy parsing
            cleaned = self._clean_expression_for_sympy(expr_text)
            
            # Attempt to parse
            parsed = self.parse_expr(cleaned, transformations='all')
            
            # Additional validation: should have mathematical atoms/operations
            atoms = parsed.atoms()
            return len(atoms) > 0 and (parsed.is_number or len(parsed.free_symbols) > 0 or parsed.has(self.sympy.Add, self.sympy.Mul))
            
        except Exception:
            return False
    
    def _clean_expression_for_sympy(self, expr_text: str) -> str:
        """
        Clean up expression text for SymPy parsing.
        Includes malformed LaTeX operator handling from Math-Verify approach.
        """
        cleaned = expr_text

        # PHASE 1: Handle malformed LaTeX operators (from Math-Verify)
        # This makes parsing more robust for MMLU/math benchmark problems
        if '\\' in cleaned or '$' in cleaned:
            # Apply LaTeX normalization for malformed operators
            cleaned = normalize_latex_for_sympy(cleaned)

        # PHASE 2: Handle common mathematical notation (existing logic)
        cleaned = cleaned.replace('°', '*pi/180')  # Convert degrees
        cleaned = cleaned.replace('×', '*')
        cleaned = cleaned.replace('÷', '/')
        cleaned = re.sub(r'(\d+)/(\d+)', r'Rational(\1,\2)', cleaned)  # Convert fractions
        cleaned = re.sub(r'equals?', '==', cleaned, flags=re.IGNORECASE)

        # Handle percentages
        cleaned = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', cleaned)

        # Handle mixed equivalence expressions (convert to equality)
        if ' or ' in cleaned or ' and ' in cleaned:
            # For expressions like "3/4 or 0.75", just take the first part for validation
            parts = re.split(r'\s+(?:or|and)\s+', cleaned)
            cleaned = parts[0]

        return cleaned
    
    def find_composite_expression_via_sympy(self, problem: str, match_text: str, match_pos: int) -> str:
        """
        Use SymPy to find mathematical composite expressions containing this component.
        This is the PRIMARY method for mathematical composite detection.

        TWO-PHASE DETECTION:
        Phase 1: LaTeX composites (from Math-Verify) - for MMLU support
        Phase 2: Plain math expressions via SymPy - existing detection
        """
        if not self.sympy_available:
            return None

        try:
            # PHASE 1: Check for LaTeX expressions first (from Math-Verify approach)
            # Critical for MMLU problems like: "What are the dimensions of $\hat{u}^t \hat{u}$?"
            latex_expressions = detect_latex_composites(problem)
            for latex_expr, start_pos, end_pos in latex_expressions:
                if start_pos <= match_pos < end_pos:
                    # Our match is within this LaTeX expression
                    if latex_expr != match_text.strip():  # It's actually composite
                        logger.debug(f"    📐 Found LaTeX composite: '{latex_expr}' (contains '{match_text}')")
                        return latex_expr

            # PHASE 2: Extract plain mathematical expressions via SymPy (existing detection)
            math_expressions = self.extract_mathematical_expressions_with_sympy(problem)

            # Check if our match is part of any extracted mathematical expression
            for expr_text, start_pos, end_pos in math_expressions:
                if start_pos <= match_pos < end_pos:
                    # Our match is within this mathematical expression
                    if expr_text != match_text.strip():  # It's actually composite
                        return expr_text

            return None

        except Exception as e:
            logger.debug(f"    Exception in find_composite_expression_via_sympy: {e}")
            return None
    
    def find_composite_expression_via_dependencies(self, problem: str, match_text: str, match_pos: int) -> str:
        """
        HYBRID approach: Use SymPy FIRST for mathematical expressions, then fall back to spaCy dependencies.
        This maintains backward compatibility while improving mathematical composite detection.
        """
        
        # PHASE 1: Try SymPy-based mathematical composite detection (PRIMARY for math)
        if self.sympy_available:
            sympy_composite = self.find_composite_expression_via_sympy(problem, match_text, match_pos)
            if sympy_composite:
                logger.debug(f"    🧮 Found mathematical composite via SymPy: '{sympy_composite}' - will handle with model")
                return sympy_composite
        
        # PHASE 2: Fall back to spaCy dependency parsing (for non-mathematical composites or SymPy failures)
        if not self.spacy_available:
            return None
        
        try:
            doc = self.nlp(problem)
            
            # Find the token(s) corresponding to our match
            target_tokens = []
            for token in doc:
                if match_pos <= token.idx < match_pos + len(match_text):
                    target_tokens.append(token)
            
            if not target_tokens:
                return None
            
            # For each target token, find the complete expression it belongs to
            for target_token in target_tokens:
                # Get the root of the expression containing this token
                expr_tokens = self._get_connected_expression_tokens(target_token)
                
                if len(expr_tokens) > 1:  # More than just the target token
                    # Sort by position and reconstruct the expression
                    expr_tokens_list = sorted(expr_tokens, key=lambda t: t.idx)
                    start_idx = expr_tokens_list[0].idx
                    end_idx = expr_tokens_list[-1].idx + len(expr_tokens_list[-1].text)
                    composite_expr = problem[start_idx:end_idx].strip()
                    
                    if composite_expr != match_text:  # It's actually composite
                        return composite_expr
            
            return None
            
        except Exception as e:
            logger.debug(f"    Exception in find_composite_expression_via_dependencies: {e}")
            return None
    
    def _get_connected_expression_tokens(self, token):
        """Get all tokens that are part of the same mathematical/temporal expression using spaCy dependencies"""
        expression_tokens = {token}
        visited = {token}
        to_process = [token]
        
        while to_process:
            current_token = to_process.pop(0)
            
            # Add children that are part of mathematical expressions (but not the whole sentence)
            for child in current_token.children:
                if child not in visited:
                    # Include mathematical operators, numbers, and immediate mathematical dependencies
                    # Using MATH_COMPOSITE_RULES configuration
                    all_math_symbols = (
                        MATH_COMPOSITE_RULES['operators'] +
                        MATH_COMPOSITE_RULES['inequalities'] +
                        MATH_COMPOSITE_RULES['special_symbols'] +
                        MATH_COMPOSITE_RULES['money_symbols']
                    )

                    if (child.pos_ in MATH_COMPOSITE_RULES['pos_tags'] or
                        child.text in all_math_symbols or
                        child.text.lower() in MATH_COMPOSITE_RULES['connecting_words'] or
                        child.text.lower() in MATH_COMPOSITE_RULES['fraction_words'] or
                        child.text.lower() in MATH_COMPOSITE_RULES['rate_indicators'] or
                        (child.dep_ in MATH_COMPOSITE_RULES['dependencies'] and child.pos_ in ['NUM', 'CCONJ']) or
                        (child.dep_ == 'cc' and child.text in ['and', 'or', '+']) or
                        (child.dep_ == 'conj' and child.pos_ == 'NUM')):
                        expression_tokens.add(child)
                        visited.add(child)
                        to_process.append(child)
            
            # For mathematical expressions, look for siblings that are also numbers/operators
            # This handles cases like "Calculate 3/4 + 1/2" where both numbers depend on "Calculate"
            if current_token.head != current_token and current_token.pos_ == 'NUM':
                for sibling in current_token.head.children:
                    if (sibling not in visited and sibling != current_token):
                        # Include other numbers and operators as siblings (using rules)
                        all_math_symbols = (
                            MATH_COMPOSITE_RULES['operators'] +
                            MATH_COMPOSITE_RULES['inequalities'] +
                            MATH_COMPOSITE_RULES['special_symbols'] +
                            MATH_COMPOSITE_RULES['money_symbols']
                        )
                        if (sibling.pos_ in MATH_COMPOSITE_RULES['pos_tags'] or
                            sibling.text in all_math_symbols or
                            (sibling.dep_ in ['nummod', 'dobj', 'compound'] and sibling.pos_ == 'NUM')):
                            expression_tokens.add(sibling)
                            visited.add(sibling)
                            to_process.append(sibling)

            # Special case for compound relationships (using rules)
            all_operators = MATH_COMPOSITE_RULES['operators'] + MATH_COMPOSITE_RULES['inequalities']
            if current_token.dep_ == 'compound' and current_token.text in all_operators:
                # The head of this operator should be included
                if current_token.head not in visited:
                    expression_tokens.add(current_token.head)
                    visited.add(current_token.head)
                    to_process.append(current_token.head)

        # Filter out tokens that are clearly not part of the mathematical expression
        # Keep only numbers, operators, and mathematical symbols (using rules)
        all_math_symbols = (
            MATH_COMPOSITE_RULES['operators'] +
            MATH_COMPOSITE_RULES['inequalities'] +
            MATH_COMPOSITE_RULES['special_symbols'] +
            MATH_COMPOSITE_RULES['money_symbols'] +
            MATH_COMPOSITE_RULES['percentage_patterns']
        )
        connecting_and_fraction_words = (
            MATH_COMPOSITE_RULES['connecting_words'] +
            MATH_COMPOSITE_RULES['fraction_words'] +
            MATH_COMPOSITE_RULES['rate_indicators']
        )

        math_expression_tokens = set()
        for tok in expression_tokens:
            if (tok.pos_ in MATH_COMPOSITE_RULES['pos_tags'] or
                tok.text in all_math_symbols or
                tok.text.lower() in connecting_and_fraction_words or
                ('/' in tok.text and tok.pos_ == 'NUM')):  # Fractions like 3/4
                math_expression_tokens.add(tok)
        
        # If we have more than just the original token, return the mathematical subset
        if len(math_expression_tokens) > 1:
            return math_expression_tokens
        else:
            return {token}  # Just return the original token if no mathematical expression found
    
    def should_skip_individual_transformation_via_deps(self, problem: str, match_text: str, match_pos: int) -> bool:
        """Check if this component is part of a composite expression using dependency parsing"""
        composite = self.find_composite_expression_via_dependencies(problem, match_text, match_pos)
        if composite:
            logger.debug(f"    🔗 Found composite via dependencies: '{composite}' - will handle with model")
            return True
        return False
    
    def collect_composite_expressions_via_deps(self, problem: str, detected_topics: Dict[str, List[Tuple[str, int]]]) -> Set[str]:
        """Collect composite expressions using dependency parsing"""
        composites = set()
        
        for topic, matches in detected_topics.items():
            for match_text, match_pos in matches:
                composite = self.find_composite_expression_via_dependencies(problem, match_text, match_pos)
                if composite:
                    composites.add(composite)
        
        return composites


class NLEntityVariationEngine:
    """
    Specialized variation engine for Natural Language entities.
    Uses spaCy for entity detection and model-based transformations.
    """
    
    def __init__(self, model_client=None):
        # Load spaCy model for NL entity detection
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except OSError:
            logger.debug("⚠️ Warning: spaCy 'en_core_web_sm' model not found.")
            logger.debug("Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            self.spacy_available = False
        
        self.model_client = model_client
    
    def find_composite_nl_expression_via_dependencies(self, problem: str, match_text: str, match_pos: int) -> str:
        """Use spaCy dependency parsing to find composite NL expressions containing this component"""
        if not self.spacy_available:
            return None
        
        try:
            doc = self.nlp(problem)
            
            # Find the token(s) corresponding to our match
            target_tokens = []
            for token in doc:
                if match_pos <= token.idx < match_pos + len(match_text):
                    target_tokens.append(token)
            
            if not target_tokens:
                return None
            
            # For each target token, find the complete NL expression it belongs to
            for target_token in target_tokens:
                # Get the root of the expression containing this token
                nl_expr_tokens = self._get_connected_nl_expression_tokens(target_token)
                
                if len(nl_expr_tokens) > 1:  # More than just the target token
                    # Sort by position and reconstruct the expression
                    nl_expr_tokens_list = sorted(nl_expr_tokens, key=lambda t: t.idx)
                    start_idx = nl_expr_tokens_list[0].idx
                    end_idx = nl_expr_tokens_list[-1].idx + len(nl_expr_tokens_list[-1].text)
                    composite_expr = problem[start_idx:end_idx].strip()
                    
                    if composite_expr != match_text:  # It's actually composite
                        return composite_expr
            
            return None
            
        except Exception as e:
            logger.debug(f"    Exception in find_composite_nl_expression_via_dependencies: {e}")
            return None
    
    def _get_connected_nl_expression_tokens(self, token):
        """Get all tokens that are part of the same NL expression using spaCy dependencies"""
        expression_tokens = {token}
        visited = {token}
        to_process = [token]
        
        while to_process:
            current_token = to_process.pop(0)
            
            # Add children that are part of NL expressions (using NL_COMPOSITE_RULES)
            for child in current_token.children:
                if child not in visited:
                    # Include coordinations, conjunctions, compounds, modifiers for named entities
                    if (child.dep_ in NL_COMPOSITE_RULES['dependencies'] or
                        (child.dep_ == 'cc' and child.text in NL_COMPOSITE_RULES['coordinators']) or
                        (child.dep_ == 'conj' and child.ent_type_ == current_token.ent_type_) or  # Same entity type
                        (child.ent_type_ != '' and child.ent_type_ == current_token.ent_type_) or  # Same named entity type
                        child.text in NL_COMPOSITE_RULES['titles']):  # NEW: Dr., Mr., etc.
                        expression_tokens.add(child)
                        visited.add(child)
                        to_process.append(child)
            
            # For named entities, look for coordinated entities (e.g., "John and Mary") - using rules
            if current_token.ent_type_ != '':
                for sibling in current_token.head.children:
                    if (sibling not in visited and sibling != current_token):
                        # Include other named entities of same type as siblings (using rules)
                        if (sibling.ent_type_ == current_token.ent_type_ or
                            sibling.text in NL_COMPOSITE_RULES['coordinators'] or
                            (sibling.dep_ in ['conj', 'appos'] and sibling.pos_ in NL_COMPOSITE_RULES['pos_tags'])):
                            expression_tokens.add(sibling)
                            visited.add(sibling)
                            to_process.append(sibling)

            # Special case for compound names (like "New York" or "John Smith") - using rules
            if current_token.dep_ in ['compound', 'flat'] or current_token.ent_type_ != '':
                # The head of this token should be included
                if (current_token.head not in visited and
                    current_token.head != current_token and
                    (current_token.head.ent_type_ == current_token.ent_type_ or
                     current_token.head.pos_ in NL_COMPOSITE_RULES['pos_tags'])):
                    expression_tokens.add(current_token.head)
                    visited.add(current_token.head)
                    to_process.append(current_token.head)

        # Filter to keep only tokens that are likely part of NL expressions (using rules)
        nl_expression_tokens = set()
        for tok in expression_tokens:
            if (tok.ent_type_ != '' or  # Named entities
                tok.pos_ in NL_COMPOSITE_RULES['pos_tags'] or
                tok.text in NL_COMPOSITE_RULES['coordinators'] or
                tok.text in NL_COMPOSITE_RULES['titles'] or  # NEW: Titles
                tok.dep_ in NL_COMPOSITE_RULES['dependencies']):
                nl_expression_tokens.add(tok)
        
        # If we have more than just the original token, return the NL subset
        if len(nl_expression_tokens) > 1:
            return nl_expression_tokens
        else:
            return {token}  # Just return the original token if no NL expression found
    
    def should_skip_individual_nl_transformation_via_deps(self, problem: str, match_text: str, match_pos: int) -> bool:
        """Check if this component is part of a composite NL expression using dependency parsing"""
        composite = self.find_composite_nl_expression_via_dependencies(problem, match_text, match_pos)
        if composite:
            logger.debug(f"    🔗 Found NL composite via dependencies: '{composite}' - will handle with model")
            return True
        return False
    
    def collect_composite_nl_expressions_via_deps(self, problem: str, nl_entities: Dict[str, List[Tuple[str, int]]]) -> set:
        """Collect composite NL expressions using dependency parsing"""
        composites = set()
        
        for entity_type, entities in nl_entities.items():
            for entity_text, entity_pos in entities:
                composite = self.find_composite_nl_expression_via_dependencies(problem, entity_text, entity_pos)
                if composite:
                    composites.add(composite)
        
        return composites

    def detect_nl_entities(self, problem: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Use spaCy to detect NL entities that can be varied.
        Returns dict with entity types and their spans.
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(problem)
        entities = defaultdict(list)
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY', 'QUANTITY']:
                entities[ent.label_].append((ent.text, ent.start_char))
        
        # Additional noun phrases (objects/concepts)
        for chunk in doc.noun_chunks:
            # Skip if already captured as named entity
            if not any(chunk.text in ent_text for ent_list in entities.values() for ent_text, _ in ent_list):
                # Filter for concrete objects/concepts
                if chunk.root.pos_ in ['NOUN'] and len(chunk.text.split()) <= 3:
                    entities['OBJECT'].append((chunk.text, chunk.start_char))
        
        # Action verbs for problem context
        action_verbs = []
        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_ in ['find', 'calculate', 'determine', 'solve', 'compute', 'measure']:
                action_verbs.append((token.text, token.idx))
        if action_verbs:
            entities['ACTION'] = action_verbs

        return dict(entities)

    def detect_nl_entities_batch(self, problems: List[str]) -> List[Dict[str, List[Tuple[str, str]]]]:
        """
        Batch detect NL entities using spaCy's efficient pipe() method.
        10-50x faster than calling detect_nl_entities() in a loop.

        Args:
            problems: List of problem texts to process

        Returns:
            List of entity dicts, one per problem (same format as detect_nl_entities)
        """
        if not self.nlp:
            return [{} for _ in problems]

        results = []

        # ✅ KEY OPTIMIZATION: Use nlp.pipe() for batch processing
        # batch_size=50 is optimal for most spaCy models
        for doc in self.nlp.pipe(problems, batch_size=50):
            entities = defaultdict(list)

            # Named entities - SAME LOGIC as detect_nl_entities()
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY', 'QUANTITY']:
                    entities[ent.label_].append((ent.text, ent.start_char))

            # Additional noun phrases (objects/concepts) - SAME LOGIC
            for chunk in doc.noun_chunks:
                # Skip if already captured as named entity
                if not any(chunk.text in ent_text for ent_list in entities.values() for ent_text, _ in ent_list):
                    # Filter for concrete objects/concepts
                    if chunk.root.pos_ in ['NOUN'] and len(chunk.text.split()) <= 3:
                        entities['OBJECT'].append((chunk.text, chunk.start_char))

            # Action verbs for problem context - SAME LOGIC
            action_verbs = []
            for token in doc:
                if token.pos_ == 'VERB' and token.lemma_ in ['find', 'calculate', 'determine', 'solve', 'compute', 'measure']:
                    action_verbs.append((token.text, token.idx))
            if action_verbs:
                entities['ACTION'] = action_verbs

            results.append(dict(entities))

        return results

    def detect_math_topics(self, problem: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Detect mathematical topics/patterns in the problem.
        Returns dict with topic types and their matches.
        """
        detected_topics = defaultdict(list)
        
        for topic, patterns in self.math_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem, re.IGNORECASE)
                for match in matches:
                    detected_topics[topic].append((match.group(), match.start()))
        
        return dict(detected_topics)
    
    def apply_math_transformations(self, problem: str, detected_topics: Dict[str, List[Tuple[str, int]]]) -> List[Dict[str, Any]]:
        """
        Apply deterministic mathematical transformations that preserve answers.
        """
        variations = []
        
        for topic, matches in detected_topics.items():
            for match_text, match_pos in matches:
                
                # DEPENDENCY-BASED COMPOSITE DETECTION: Skip if part of composite expression
                if hasattr(self, 'should_skip_individual_transformation_via_deps') and self.should_skip_individual_transformation_via_deps(problem, match_text, match_pos):
                    continue
                
                # Apply relevant transformations based on detected topic
                if topic == 'fractions':
                    # Try fraction to decimal
                    if match_text in self.math_transformations['fraction_to_decimal']:
                        new_text = problem.replace(
                            match_text, 
                            self.math_transformations['fraction_to_decimal'][match_text], 
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'fraction_to_decimal',
                            'original_component': match_text,
                            'new_component': self.math_transformations['fraction_to_decimal'][match_text],
                            'topic': topic
                        })
                    
                    # Try fraction to percentage
                    if match_text in self.math_transformations['fraction_to_percentage']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['fraction_to_percentage'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'fraction_to_percentage',
                            'original_component': match_text,
                            'new_component': self.math_transformations['fraction_to_percentage'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'decimals':
                    # Try decimal to fraction
                    if match_text in self.math_transformations['decimal_to_fraction']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['decimal_to_fraction'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'decimal_to_fraction', 
                            'original_component': match_text,
                            'new_component': self.math_transformations['decimal_to_fraction'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'number_words':
                    # Try word to number
                    if match_text.lower() in self.math_transformations['word_to_number']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['word_to_number'][match_text.lower()],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'word_to_number',
                            'original_component': match_text,
                            'new_component': self.math_transformations['word_to_number'][match_text.lower()],
                            'topic': topic
                        })
                
                elif topic == 'trigonometry':
                    # Try trigonometric equivalents
                    if match_text in self.math_transformations['trigonometric_equivalents']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['trigonometric_equivalents'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'trigonometric_equivalent',
                            'original_component': match_text,
                            'new_component': self.math_transformations['trigonometric_equivalents'][match_text],
                            'topic': topic
                        })
                
                elif topic == 'algebra':
                    # Try algebraic equivalents
                    if match_text in self.math_transformations['algebraic_equivalents']:
                        new_text = problem.replace(
                            match_text,
                            self.math_transformations['algebraic_equivalents'][match_text],
                            1
                        )
                        variations.append({
                            'modified_problem': new_text,
                            'transformation_type': 'algebraic_equivalent',
                            'original_component': match_text,
                            'new_component': self.math_transformations['algebraic_equivalents'][match_text],
                            'topic': topic
                        })
        
        return variations
    
    def apply_nl_variations(self, problem: str, nl_entities: Dict[str, List[Tuple[str, int]]]) -> List[Dict[str, Any]]:
        """
        Apply model-based variations to detected NL entities.
        """
        if not self.model_client:
            logger.debug("⚠️ No model client available for NL variations")
            return []
        
        variations = []
        
        for entity_type, entities in nl_entities.items():
            for entity_text, entity_pos in entities:
                
                # DEPENDENCY-BASED COMPOSITE DETECTION: Skip if part of composite expression
                if self.should_skip_individual_nl_transformation_via_deps(problem, entity_text, entity_pos):
                    continue
                
                # Generate variations based on entity type
                if entity_type == 'PERSON':
                    variations_prompt = f"""
                    Replace the person name "{entity_text}" with semantically equivalent alternatives that preserve the problem's meaning.
                    
                    Original problem: {problem}
                    
                    Generate 3 variations where ONLY "{entity_text}" is replaced:
                    - Generic person reference (a person, someone, a student, etc.)
                    - Different name (keeping same gender context if relevant)
                    - Role-based reference (a worker, a customer, etc.)
                    
                    Keep everything else EXACTLY the same.
                    
                    Format:
                    1. [complete modified problem]
                    2. [complete modified problem]  
                    3. [complete modified problem]
                    """
                
                elif entity_type == 'OBJECT':
                    variations_prompt = f"""
                    Replace the object "{entity_text}" with semantically equivalent alternatives.
                    
                    Original problem: {problem}
                    
                    Generate 3 variations where ONLY "{entity_text}" is replaced:
                    - Generic equivalent (items, objects, things)
                    - Specific alternative of same category
                    - Different phrasing for same concept
                    
                    Keep everything else EXACTLY the same.
                    
                    Format:
                    1. [complete modified problem]
                    2. [complete modified problem]
                    3. [complete modified problem]
                    """
                
                elif entity_type == 'ACTION':
                    variations_prompt = f"""
                    Replace the action verb "{entity_text}" with equivalent alternatives.
                    
                    Original problem: {problem}
                    
                    Generate 3 variations where ONLY "{entity_text}" is replaced:
                    - Synonymous action verb
                    - More formal/informal version
                    - Different phrasing with same meaning
                    
                    Keep everything else EXACTLY the same.
                    
                    Format:
                    1. [complete modified problem]
                    2. [complete modified problem]
                    3. [complete modified problem]
                    """
                
                else:
                    continue  # Skip entity types we don't handle
                
                # Get model response  
                try:
                    # Create proper system and user prompts
                    system_prompt = """You are an expert at generating sophisticated natural language variations that preserve exact meaning, answers, and question intent. 

SOPHISTICATED TRANSFORMATION RULES:
- Create complex, descriptive alternatives (not simple synonyms)
- TIME: "5pm" → "the time when the clock reads five o'clock in the evening"
- ENTITIES: "rectangle" → "a four-sided geometric figure with right angles"
- QUANTITIES: "3/4" → "three quarters of the total amount"
- ACTIONS: "Calculate" → "Determine the numerical value of"
- Use creative language that requires deeper comprehension
- Avoid simple word swaps - make transformations sophisticated

CRITICAL: Do NOT change what is being asked or the fundamental problem type."""
                    response = self.model_client.get_model_response([system_prompt], [variations_prompt])
                    if isinstance(response, list):
                        response = response[0]
                    
                    # Parse variations from response
                    parsed_variations = self._parse_nl_variations(response, entity_text, entity_type, problem)
                    variations.extend(parsed_variations)
                
                except Exception as e:
                    logger.debug(f"Error generating NL variations for {entity_text}: {e}")
                    continue
        
        return variations
    
    def _parse_nl_variations(self, response: str, original_entity: str, entity_type: str, original_problem: str) -> List[Dict[str, Any]]:
        """Parse model response to extract NL variations and identify replacement components."""
        variations = []
        
        # Simple parsing - look for numbered lines
        lines = clean_model_response(response).split('\n')
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                # Extract the problem text
                problem_text = re.sub(r'^\d+\.\s*', '', line.strip())
                if problem_text and problem_text != original_problem:
                    # Extract the actual replacement component by comparing original and modified
                    new_component = self._extract_replacement_component(
                        original_problem, problem_text, original_entity
                    )
                    
                    variations.append({
                        'modified_problem': clean_model_response(problem_text),
                        'transformation_type': f'nl_{entity_type.lower()}_variation',
                        'original_component': original_entity,
                        'new_component': new_component,
                        'topic': entity_type
                    })
        
        return variations
    
    def _extract_replacement_component(self, original_problem: str, modified_problem: str, original_component: str) -> str:
        """
        Extract the actual replacement component by comparing original and modified problems.
        This identifies what text replaced the original component.
        """
        try:
            # Find all occurrences of the original component in the original problem
            original_positions = []
            start = 0
            while True:
                pos = original_problem.find(original_component, start)
                if pos == -1:
                    break
                original_positions.append((pos, pos + len(original_component)))
                start = pos + 1
            
            if not original_positions:
                return "model_generated"  # Fallback if component not found
            
            # For each occurrence, find what it became in the modified problem
            for start_pos, end_pos in original_positions:
                # Look at the context around this position in both problems
                prefix_len = min(20, start_pos)  # Look at up to 20 chars before
                suffix_len = min(20, len(original_problem) - end_pos)  # Look at up to 20 chars after
                
                prefix = original_problem[start_pos - prefix_len:start_pos]
                suffix = original_problem[end_pos:end_pos + suffix_len]
                
                # Find this context in the modified problem
                prefix_pos = modified_problem.find(prefix)
                if prefix_pos != -1:
                    # Extract what's between the prefix and suffix in the modified problem
                    after_prefix = prefix_pos + len(prefix)
                    suffix_pos = modified_problem.find(suffix, after_prefix)
                    
                    if suffix_pos != -1:
                        replacement = modified_problem[after_prefix:suffix_pos].strip()
                        if replacement and replacement != original_component:
                            return replacement
            
            # If context matching fails, try simpler approach: find longest common subsequences
            # and identify the different part
            return self._find_replacement_by_diff(original_problem, modified_problem, original_component)
            
        except Exception as e:
            logger.debug(f"    Error extracting replacement component: {e}")
            return "model_generated"  # Fallback
    
    def _find_replacement_by_diff(self, original: str, modified: str, original_component: str) -> str:
        """Find replacement using simple difference analysis"""
        try:
            # Split into words for easier comparison
            orig_words = original.split()
            mod_words = modified.split()
            
            # Find the original component words
            orig_comp_words = original_component.split()
            
            if not orig_comp_words:
                return "model_generated"
            
            # Look for the position where original component appears
            for i in range(len(orig_words) - len(orig_comp_words) + 1):
                if orig_words[i:i+len(orig_comp_words)] == orig_comp_words:
                    # Found the original component at position i
                    # Check what's at this position in modified
                    if i < len(mod_words):
                        # Determine how many words the replacement spans
                        # This is tricky without alignment, so take a reasonable guess
                        
                        # Simple heuristic: if the sentences are roughly the same length,
                        # assume 1:1 word replacement
                        if abs(len(orig_words) - len(mod_words)) <= 2:
                            if i + len(orig_comp_words) <= len(mod_words):
                                replacement_words = mod_words[i:i+len(orig_comp_words)]
                                return " ".join(replacement_words)
                        
                        # For different length sentences, try to find a reasonable replacement
                        # Look for a sequence that makes sense
                        for length in range(1, min(4, len(mod_words) - i + 1)):  # Try 1-3 word replacements
                            replacement_words = mod_words[i:i+length]
                            replacement = " ".join(replacement_words)
                            if replacement != original_component:
                                return replacement
            
            return "model_generated"  # Fallback if no replacement found
            
        except Exception:
            return "model_generated"  # Fallback


class ComprehensiveVariationEngine:
    """
    Main variation engine that orchestrates domain-specific engines.
    Combines Math and NL entity variation engines for comprehensive coverage.
    """
    
    def __init__(self, model_client=None, judge_model_client=None):
        self.math_engine = MathDomainVariationEngine()
        self.nl_engine = NLEntityVariationEngine(model_client)
        self.temporal_engine = TemporalDomainVariationEngine()
        self.model_client = model_client
        self.judge_model_client = judge_model_client or model_client  # Use judge if provided, else variation model
        
        # Initialize consistency enhancer
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            self.consistency_enhancer = VariationConsistencyEnhancer(nlp)
            self.consistency_enhancement_enabled = True
            logger.debug("✅ Consistency enhancement enabled")
        except:
            self.consistency_enhancer = None
            self.consistency_enhancement_enabled = False
            logger.debug("⚠️ Consistency enhancement disabled (spaCy not available)")
        
        # Report model client status
        if model_client:
            logger.debug("✅ Model client provided for NL variations")
        else:
            logger.debug("⚠️ No model client - NL variations will be limited")
    
    
    def generate_comprehensive_variations(self, problem: str, 
                                        min_near_transfer: int = 3, max_near_transfer: int = 8,
                                        min_far_transfer: int = 2, max_far_transfer: int = 5,
                                        enable_optimization: bool = False,  # DISABLED by default 
                                        enable_priority_scoring: bool = False,  # NEW: Priority switch
                                        max_variations: int = None) -> List[Dict[str, Any]]:
        """
        Generate comprehensive variations using working pattern detection engines.
        """
        logger.debug(f"Generating variations for: {problem[:100]}...")
        
        all_variations = []
        
        # 1. Detect NL entities
        logger.debug("  Detecting NL entities...")
        nl_entities = self.nl_engine.detect_nl_entities(problem)
        logger.debug(f"    Found NL entities: {dict(nl_entities)}")
        
        # 2. Detect mathematical topics
        logger.debug("  Detecting math topics...")
        math_topics = self.math_engine.detect_math_topics(problem)
        logger.debug(f"    Found math topics: {dict(math_topics)}")
        
        # 3. Detect temporal topics
        logger.debug("  Detecting temporal topics...")
        temporal_topics = self.temporal_engine.detect_temporal_topics(problem)
        logger.debug(f"    Found temporal topics: {dict(temporal_topics)}")
        
        # 4. Generate unitary variations (single component changes)
        logger.debug("  Applying unitary transformations...")
        
        # Math transformations
        math_variations = self.math_engine.apply_math_transformations(problem, math_topics)
        all_variations.extend(math_variations)
        logger.debug(f"    Generated {len(math_variations)} math variations")
        
        # Temporal transformations
        temporal_variations = self.temporal_engine.apply_temporal_transformations(problem, temporal_topics)
        all_variations.extend(temporal_variations)
        logger.debug(f"    Generated {len(temporal_variations)} temporal variations")
        
        # NL variations (uses model client if available)
        nl_variations = self.nl_engine.apply_nl_variations(problem, nl_entities)
        all_variations.extend(nl_variations)
        if self.model_client:
            logger.debug(f"    Generated {len(nl_variations)} NL variations using model client")
        else:
            logger.debug(f"    Generated {len(nl_variations)} NL variations (no model client)")
        
        # 5. Generate model-guided variations for ALL domains (in addition to static)
        if self.model_client:
            logger.debug("  Generating model-guided variations for all domains...")
            
            # Collect ALL detected candidates from all domains
            all_candidates = []
            
            # Math candidates
            for topic, matches in math_topics.items():
                for match_text, match_pos in matches:
                    all_candidates.append({'text': match_text, 'pos': match_pos, 'domain': 'math', 'topic': topic})
            
            # Temporal candidates  
            for topic, matches in temporal_topics.items():
                for match_text, match_pos in matches:
                    all_candidates.append({'text': match_text, 'pos': match_pos, 'domain': 'temporal', 'topic': topic})
            
            # NL candidates
            for entity_type, matches in nl_entities.items():
                for match_text, match_pos in matches:
                    all_candidates.append({'text': match_text, 'pos': match_pos, 'domain': 'nl', 'topic': entity_type})
            
            if all_candidates:
                # Generate NEAR TRANSFER variations (surface-level changes)
                near_variations = self.generate_near_transfer_variations(
                    problem, all_candidates, min_near_transfer, max_near_transfer)
                all_variations.extend(near_variations)
                logger.debug(f"    Generated {len(near_variations)} near-transfer variations")
                
                # Generate FAR TRANSFER variations (complex, twisted changes)  
                far_variations = self.generate_far_transfer_variations(
                    problem, all_candidates, min_far_transfer, max_far_transfer)
                all_variations.extend(far_variations)
                logger.debug(f"    Generated {len(far_variations)} far-transfer variations")
        
        # 6. Generate combination variations (multiple component changes)
        logger.debug("  Generating combination variations...")
        combination_variations = self.generate_combination_variations(
            problem, math_variations + temporal_variations, nl_variations
        )
        all_variations.extend(combination_variations)
        logger.debug(f"    Generated {len(combination_variations)} combination variations")
        
        logger.debug(f"  Total variations generated: {len(all_variations)}")
        
        # 7. Generate generic domain-invariant variations
        logger.debug("  Generating generic variations...")
        generic_variations = self.generate_generic_variations(problem)
        all_variations.extend(generic_variations)
        logger.debug(f"    Generated {len(generic_variations)} generic variations")
        
        # 8. Ensure minimum combinations per problem
        min_combinations = 4  # User requirement
        if len(combination_variations) < min_combinations:
            logger.debug(f"  Ensuring minimum {min_combinations} combinations...")
            additional_combinations = self.generate_model_guided_combinations(
                problem, all_variations, min_combinations - len(combination_variations)
            )
            all_variations.extend(additional_combinations)
            logger.debug(f"    Added {len(additional_combinations)} model-guided combinations")

        # 9. Add comprehensive metadata to ALL variations (ensuring all details are saved)
        for variation in all_variations:
            # Essential metadata
            if 'original_problem' not in variation:
                variation['original_problem'] = problem
            if 'detection_method' not in variation:
                variation['detection_method'] = 'spacy_pattern_detection'
            if 'generation_method' not in variation:
                variation['generation_method'] = 'domain_engines'
            
            # Confidence level
            if 'confidence' not in variation:
                if variation.get('transformation_type', '').startswith('model_'):
                    variation['confidence'] = 'model_generated'
                else:
                    variation['confidence'] = 'pattern_based'
            
            # Combination order details
            if 'combination_order' not in variation:
                if 'combination' in variation.get('transformation_type', '').lower():
                    variation['combination_order'] = variation.get('combination_order', 2)  # Default for combinations
                else:
                    variation['combination_order'] = 1  # Unitary transformation
            
            # Topic/domain information
            if 'topic' not in variation:
                variation['topic'] = variation.get('topic', 'general')
            
            # Ensure transformation type is present
            if 'transformation_type' not in variation:
                variation['transformation_type'] = 'unknown'
            
            # Ensure component information is present
            if 'original_component' not in variation:
                variation['original_component'] = 'N/A'
            if 'new_component' not in variation:
                variation['new_component'] = 'N/A'

        # 10. Apply consistency enhancement if available
        if self.consistency_enhancement_enabled:
            enhanced_variations = []
            enhancement_count = 0
            
            for variation in all_variations:
                enhanced_variation = self._enhance_single_variation(problem, variation)
                enhanced_variations.append(enhanced_variation)
                
                # Count if enhancement was applied
                if enhanced_variation.get('modified_problem') != variation.get('modified_problem'):
                    enhancement_count += 1
            
            logger.debug(f"  Consistency enhancements applied: {enhancement_count}")
            all_variations = enhanced_variations  # Use enhanced variations for further processing
        
        # 11. Apply variation optimization and priority-based selection (if enabled)
        if enable_optimization:
            logger.debug("  Applying variation optimization and priority selection...")
            optimized_variations = self._optimize_and_prioritize_variations(
                problem, all_variations, max_variations, enable_priority_scoring
            )
            logger.debug(f"    Selected {len(optimized_variations)} high-priority variations from {len(all_variations)} candidates")
            return optimized_variations
        else:
            logger.debug("  Optimization disabled - returning all variations")
            return all_variations
    
    def generate_debugging_focused_variations(self, problem: str, 
                                            enable_unitary_transformations: bool = False,
                                            enable_candidate_combinations: bool = True,
                                            combination_sizes: List[int] = [2, 3]) -> Dict[str, List[Dict[str, Any]]]:
        """
        NEW: Generate variations organized by debugging purpose rather than just domain.
        Tests specific cognitive capabilities that models commonly struggle with.
        
        Args:
            problem: The input problem to generate variations for
            enable_unitary_transformations: Include single-candidate transformations (default: False)
            enable_candidate_combinations: Include multi-candidate combinations (default: True)
            combination_sizes: List of combination sizes to generate (default: [2, 3])
        
        Returns: Dictionary mapping capability names to variation lists
        """
        logger.debug("🔍 Generating debugging-focused variations...")
        
        # Get candidate combinations instead of unitary transformations
        if enable_candidate_combinations:
            logger.debug("  Using candidate combination approach...")
            domain_variations = self._get_combination_variations_for_debugging(
                problem, combination_sizes, enable_unitary_transformations
            )
        else:
            # Fallback to original approach
            domain_variations = self._get_domain_variations_for_debugging(problem)
        
        # Then organize them by debugging purpose (assign each variation to ONE capability only)
        debugging_variations = {
            'numerical_robustness': [],
            'format_dependency': [],
            'context_preservation': [],
            'unit_sensitivity': [],
            'linguistic_flexibility': [],
            'temporal_reasoning': [],
            'mathematical_equivalence': [],
            'integration_capability': []
        }
        
        # Assign each variation to exactly one debugging capability
        all_variations_list = []
        for domain_variations_list in domain_variations.values():
            all_variations_list.extend(domain_variations_list)
        
        for variation in all_variations_list:
            assigned_capability = self._assign_primary_debugging_capability(variation)
            if assigned_capability and assigned_capability in debugging_variations:
                # Add debugging metadata
                variation['debugging_capability'] = assigned_capability
                variation['debugging_purpose'] = self._get_debugging_purpose(assigned_capability)
                variation['debugging_rationale'] = self._get_debugging_rationale(assigned_capability)
                debugging_variations[assigned_capability].append(variation)
        
        # Add debugging metadata to each variation
        total_variations = 0
        for capability, variations in debugging_variations.items():
            for variation in variations:
                variation['debugging_capability'] = capability
                variation['debugging_purpose'] = self._get_capability_description(capability)
            total_variations += len(variations)
        
        logger.debug(f"   Generated {total_variations} variations across {len(debugging_variations)} debugging capabilities")
        return debugging_variations
    
    def _get_domain_variations_for_debugging(self, problem: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get variations from existing domain engines for debugging reorganization"""
        
        # Reuse existing detection logic
        logger.debug("  Detecting entities and topics for debugging...")
        nl_entities = self.nl_engine.detect_nl_entities(problem)
        math_topics = self.math_engine.detect_math_topics(problem) 
        temporal_topics = self.temporal_engine.detect_temporal_topics(problem)
        
        logger.debug(f"    Found NL entities: {dict(nl_entities)}")
        logger.debug(f"    Found math topics: {dict(math_topics)}")
        logger.debug(f"    Found temporal topics: {dict(temporal_topics)}")
        
        # Generate variations using existing engines
        domain_variations = {
            'math': self.math_engine.apply_math_transformations(problem, math_topics),
            'temporal': self.temporal_engine.apply_temporal_transformations(problem, temporal_topics),
            'nl': []  # Will add NL variations if model client available
        }
        
        # Add NL variations if we have model client
        if self.model_client and nl_entities:
            domain_variations['nl'] = self.nl_engine.generate_nl_variations(problem, nl_entities, self.model_client)
        
        return domain_variations
    
    def _get_combination_variations_for_debugging(self, problem: str, 
                                                combination_sizes: List[int],
                                                include_unitary: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate variations using intelligent candidate combinations.
        Starts with max combinations, uses model to prioritize, generates meaningful variations.
        """
        from itertools import combinations
        
        logger.debug("  Detecting all candidates for combination...")
        
        # Detect all available candidates
        nl_entities = self.nl_engine.detect_nl_entities(problem)
        math_topics = self.math_engine.detect_math_topics(problem)
        temporal_topics = self.temporal_engine.detect_temporal_topics(problem)
        
        logger.debug(f"    Found NL entities: {dict(nl_entities)}")
        logger.debug(f"    Found math topics: {dict(math_topics)}")
        logger.debug(f"    Found temporal topics: {dict(temporal_topics)}")
        
        # Collect all transformable candidates
        all_candidates = []
        for topic, candidates in math_topics.items():
            for candidate, pos in candidates:
                all_candidates.append({'text': candidate, 'pos': pos, 'domain': 'math', 'topic': topic})
        for topic, candidates in temporal_topics.items():
            for candidate, pos in candidates:
                all_candidates.append({'text': candidate, 'pos': pos, 'domain': 'temporal', 'topic': topic})
        for entity_type, entities in nl_entities.items():
            for entity_text, pos in entities:
                all_candidates.append({'text': entity_text, 'pos': pos, 'domain': 'nl', 'topic': entity_type})
        
        logger.debug(f"    Total transformable candidates: {len(all_candidates)}")
        
        if not all_candidates or not self.model_client:
            logger.debug("    ⚠️ No candidates or no model client - falling back to basic combinations")
            return {'math': [], 'temporal': [], 'nl': [], 'combinations': []}
        
        # Calculate intelligent combination sizes (start with max, work down)
        max_candidates = min(len(all_candidates), 5)  # Cap at 5 for practical reasons
        intelligent_sizes = list(range(max_candidates, 1, -1))  # [5, 4, 3, 2]
        
        logger.debug(f"    Using intelligent combination sizes: {intelligent_sizes}")
        
        # Calculate max combinations to avoid explosion
        max_total_combinations = min(50, len(all_candidates) * 3)  # Proportional limit
        
        # Step 1: Get model to prioritize combinations
        priority_combinations = self._get_prioritized_combinations(
            problem, all_candidates, intelligent_sizes, max_total_combinations
        )
        
        # Step 2: Generate variations for prioritized combinations
        domain_variations = {'combinations': []}
        
        for i, combo_data in enumerate(priority_combinations):
            if i >= max_total_combinations:
                break
                
            variation = self._generate_meaningful_variation(problem, combo_data)
            if variation:
                domain_variations['combinations'].append(variation)
        
        logger.debug(f"    Generated {len(domain_variations['combinations'])} intelligent combination variations")
        
        return domain_variations
    
    def _get_prioritized_combinations(self, problem: str, all_candidates: List[Dict], 
                                     sizes: List[int], max_combinations: int) -> List[Dict]:
        """Use model to prioritize which candidate combinations are most impactful."""
        from itertools import combinations
        
        logger.debug(f"    Step 1: Getting model to prioritize combinations...")
        
        # Create candidate summary for the model
        candidate_summary = []
        for i, cand in enumerate(all_candidates):
            candidate_summary.append(f"{i+1}. '{cand['text']}' ({cand['domain']}:{cand['topic']})")
        
        system_prompt = """You are an expert at identifying the most impactful candidate combinations for creating meaningful question variations.

TASK: Given a problem and candidates, identify the MOST IMPACTFUL combinations that would create the most meaningful and diverse variations.

PRIORITIZATION CRITERIA:
1. IMPACT: Combinations that significantly change the problem's character
2. DIVERSITY: Combinations that test different cognitive capabilities  
3. MEANINGFULNESS: Combinations that create substantially different questions
4. AVOID: Trivial changes or redundant combinations

OUTPUT FORMAT:
Return EXACTLY the top combinations in this format:
[1,3,5] - Brief reason why this combination is impactful
[2,4] - Brief reason why this combination is impactful
[1,2,3,4] - Brief reason why this combination is impactful

RULES:
- Start with larger combinations (4-5 candidates) then smaller ones
- Focus on combinations that cross domains (math+temporal+nl)
- Avoid combinations that would create minimal changes
- Maximum {max_combinations} combinations total"""

        user_prompt = f"""Problem: {problem}

Available candidates:
{chr(10).join(candidate_summary)}

Identify the top {max_combinations} most impactful candidate combinations, starting with larger combinations (4-5 candidates) down to smaller ones (2-3 candidates).

Focus on combinations that would create the most meaningful and diverse question variations."""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(user_prompt, system_prompt))
            
            # Parse the model's priority response
            priority_combinations = self._parse_priority_response(response, all_candidates)
            logger.debug(f"    Model prioritized {len(priority_combinations)} combinations")
            
            return priority_combinations
            
        except Exception as e:
            logger.debug(f"    ⚠️ Model prioritization failed: {e}, using fallback")
            return self._fallback_combinations(all_candidates, sizes, max_combinations)
    
    def _parse_priority_response(self, response: str, all_candidates: List[Dict]) -> List[Dict]:
        """Parse model's priority response into combination data."""
        import re
        
        combinations = []
        lines = clean_model_response(response).split('\n')
        
        for line in lines:
            # Look for pattern like [1,3,5] - reason
            match = re.search(r'\[([0-9,\s]+)\]\s*-?\s*(.*)', line.strip())
            if match:
                indices_str = match.group(1)
                reason = match.group(2).strip()
                
                try:
                    # Parse indices (convert from 1-based to 0-based)
                    indices = [int(x.strip()) - 1 for x in indices_str.split(',')]
                    
                    # Validate indices
                    valid_indices = [i for i in indices if 0 <= i < len(all_candidates)]
                    
                    if len(valid_indices) >= 2:  # At least 2 candidates
                        combo_candidates = [all_candidates[i] for i in valid_indices]
                        combinations.append({
                            'candidates': combo_candidates,
                            'size': len(valid_indices),
                            'reason': reason,
                            'priority_score': len(combinations) + 1  # Higher priority = lower score
                        })
                except:
                    continue
        
        return combinations
    
    def _fallback_combinations(self, all_candidates: List[Dict], sizes: List[int], max_combinations: int) -> List[Dict]:
        """Fallback combination selection when model fails."""
        from itertools import combinations
        
        fallback_combinations = []
        count = 0
        
        for size in sizes:
            if count >= max_combinations:
                break
                
            for combo in combinations(all_candidates, size):
                if count >= max_combinations:
                    break
                    
                fallback_combinations.append({
                    'candidates': list(combo),
                    'size': size,
                    'reason': f'Automatic {size}-candidate combination',
                    'priority_score': count + 1
                })
                count += 1
        
        return fallback_combinations
    
    def _generate_meaningful_variation(self, problem: str, combo_data: Dict) -> Dict[str, Any]:
        """Generate a meaningful variation for a prioritized combination."""
        candidates = combo_data['candidates']
        reason = combo_data['reason']
        
        # Create detailed candidate info for the model
        candidate_details = []
        for cand in candidates:
            candidate_details.append(f"'{cand['text']}' ({cand['domain']}:{cand['topic']})")
        
        system_prompt = """You are an expert at creating meaningful question variations by transforming multiple components simultaneously.

CRITICAL REQUIREMENTS:
1. PRESERVE exact numerical answer and question intent
2. CREATE meaningful, substantial changes (not trivial format changes)
3. TRANSFORM all specified candidates in creative, sophisticated ways
4. ENSURE the variation tests different cognitive capabilities
5. MAINTAIN mathematical/logical validity

TRANSFORMATION QUALITY:
- GOOD: "3/4" → "three quarters of the total", "John walks" → "John traverses on foot"
- BAD: "3/4" → "0.75" (too trivial), changing core question logic

OUTPUT: Return ONLY the modified question in plain text, no formatting."""

        user_prompt = f"""Original problem: {problem}

Transform these candidates simultaneously:
{', '.join(candidate_details)}

Reason for this combination: {reason}

Create ONE meaningful variation that transforms all these candidates in sophisticated, creative ways while preserving the exact answer and question intent."""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(user_prompt, system_prompt))
            
            cleaned_response = clean_model_response(response)
            
            if cleaned_response and is_valid_question(cleaned_response):
                return {
                    'original_problem': problem,
                    'modified_problem': cleaned_response,
                    'transformation_type': f'intelligent_combination_{combo_data["size"]}way',
                    'candidates_transformed': [c['text'] for c in candidates],
                    'combination_reason': reason,
                    'combination_size': combo_data['size'],
                    'priority_score': combo_data['priority_score'],
                    'confidence': 'model_generated',
                    'combination_order': combo_data['size']
                }
        except Exception as e:
            logger.debug(f"      ⚠️ Failed to generate variation for combination: {e}")
        
        return None
    
    def _assign_primary_debugging_capability(self, variation: Dict) -> str:
        """Assign a variation to exactly one debugging capability based on its characteristics."""
        transform_type = variation.get('transformation_type', '')
        candidates = variation.get('candidates_transformed', [])
        combination_size = variation.get('combination_size', 1)
        
        # Priority-based assignment (each variation goes to ONE capability)
        
        # 1. Integration capability for multi-candidate combinations
        if combination_size >= 4 or 'intelligent_combination' in transform_type:
            return 'integration_capability'
        
        # 2. Mathematical equivalence for math transformations
        if any(domain in transform_type for domain in ['math', 'fraction', 'decimal', 'geometry']):
            return 'mathematical_equivalence'
        
        # 3. Temporal reasoning for time-related transformations
        if any(domain in transform_type for domain in ['temporal', 'time', 'duration', 'schedule']):
            return 'temporal_reasoning'
        
        # 4. Format dependency for format changes
        if any(keyword in transform_type for keyword in ['format', '12hr', '24hr', 'decimal', 'percentage']):
            return 'format_dependency'
        
        # 5. Linguistic flexibility for NL transformations
        if any(domain in transform_type for domain in ['nl_', 'entity', 'action', 'object']):
            return 'linguistic_flexibility'
        
        # 6. Numerical robustness for number-related changes
        if any(keyword in str(candidates).lower() for keyword in ['number', 'quantity', 'fraction', 'decimal']):
            return 'numerical_robustness'
        
        # 7. Unit sensitivity for unit changes
        if any(keyword in str(candidates).lower() for keyword in ['hour', 'minute', 'second', 'cm', 'mm', 'mile', 'km']):
            return 'unit_sensitivity'
        
        # 8. Context preservation as fallback
        return 'context_preservation'
    
    def _get_debugging_purpose(self, capability: str) -> str:
        """Get the purpose description for a debugging capability."""
        purposes = {
            'numerical_robustness': 'Tests model handling of different numerical representations and formats',
            'format_dependency': 'Tests whether model depends too heavily on specific input formats',
            'context_preservation': 'Tests model ability to maintain meaning through context changes',
            'unit_sensitivity': 'Tests model awareness and handling of measurement units',
            'linguistic_flexibility': 'Tests model robustness to linguistic variations and synonyms',
            'temporal_reasoning': 'Tests model understanding of time-related concepts and logic',
            'mathematical_equivalence': 'Tests model recognition of mathematically equivalent expressions',
            'integration_capability': 'Tests model ability to handle multiple simultaneous changes'
        }
        return purposes.get(capability, 'Tests general reasoning capability')
    
    def _get_debugging_rationale(self, capability: str) -> str:
        """Get the rationale for why this capability is being tested."""
        rationales = {
            'numerical_robustness': 'Identifies if model struggles with different number formats',
            'format_dependency': 'Reveals model over-reliance on specific formatting patterns',
            'context_preservation': 'Exposes model sensitivity to context and phrasing changes',
            'unit_sensitivity': 'Tests model awareness of measurement units and conversions',
            'linguistic_flexibility': 'Evaluates model handling of linguistic variations',
            'temporal_reasoning': 'Assesses model understanding of temporal concepts',
            'mathematical_equivalence': 'Tests model recognition of equivalent mathematical forms',
            'integration_capability': 'Evaluates model ability to handle complex multi-faceted changes'
        }
        return rationales.get(capability, 'General reasoning assessment')
    
    def _apply_candidate_combination(self, problem: str, candidate_combo: tuple) -> Dict[str, Any]:
        """Apply transformations to a combination of candidates"""
        modified_problem = problem
        transformation_details = []
        domains_involved = set()
        
        # Sort candidates by position (right to left to preserve positions)
        sorted_candidates = sorted(candidate_combo, key=lambda x: x['position'], reverse=True)
        
        for candidate in sorted_candidates:
            # Get appropriate transformation for this candidate
            transformation = self._get_transformation_for_candidate(candidate)
            
            if transformation:
                # Apply the transformation
                old_text = candidate['text']
                new_text = transformation['new_text']
                
                # Replace in problem (right to left preserves positions)
                start_pos = candidate['position']
                end_pos = start_pos + len(old_text)
                modified_problem = modified_problem[:start_pos] + new_text + modified_problem[end_pos:]
                
                transformation_details.append({
                    'original': old_text,
                    'new': new_text,
                    'domain': candidate['domain'],
                    'type': transformation['type']
                })
                domains_involved.add(candidate['domain'])
        
        if not transformation_details:
            return None
        
        # Create variation metadata
        variation = {
            'original_problem': problem,
            'modified_problem': modified_problem,
            'transformation_type': f"combination_{len(candidate_combo)}way_" + "_".join(sorted(domains_involved)),
            'combination_order': len(candidate_combo),
            'domains_involved': list(domains_involved),
            'transformations': transformation_details,
            'confidence': 'pattern_based',
            'transfer_type': 'near' if len(candidate_combo) <= 2 else 'far'
        }
        
        return variation
    
    def _get_transformation_for_candidate(self, candidate: Dict) -> Dict[str, Any]:
        """Get appropriate transformation for a specific candidate"""
        domain = candidate['domain']
        text = candidate['text']
        topic = candidate['topic']
        
        if domain == 'math':
            return self._get_math_transformation(text, topic)
        elif domain == 'temporal':
            return self._get_temporal_transformation(text, topic)
        elif domain == 'nl':
            return self._get_nl_transformation(text, topic)
        
        return None
    
    def _get_math_transformation(self, text: str, topic: str) -> Dict[str, Any]:
        """Get math transformation for a candidate"""
        # Reuse existing math transformation logic
        if topic == 'fractions':
            if '/' in text:
                try:
                    # Try decimal conversion
                    parts = text.split('/')
                    if len(parts) == 2:
                        decimal_val = float(parts[0]) / float(parts[1])
                        return {'new_text': str(decimal_val), 'type': 'fraction_to_decimal'}
                except:
                    pass
        
        elif topic == 'geometry':
            geometry_equivalents = {
                'rectangle': 'rectangular shape',
                'area': 'total area',
                'total area': 'area'
            }
            if text in geometry_equivalents:
                return {'new_text': geometry_equivalents[text], 'type': 'geometry_equivalent'}
        
        return None
    
    def _get_temporal_transformation(self, text: str, topic: str) -> Dict[str, Any]:
        """Get temporal transformation for a candidate"""
        if topic == 'durations':
            duration_conversions = {
                '2 hours': '120 minutes',
                '1 hour': '60 minutes', 
                '3 hours': '180 minutes',
                '4 hours': '240 minutes',
                '5 hours': '300 minutes'
            }
            if text in duration_conversions:
                return {'new_text': duration_conversions[text], 'type': 'duration_to_minutes'}
        
        elif topic in ['times_12hr', 'times_24hr']:
            time_conversions = {
                '3:00 PM': '15:00',
                '2:00 PM': '14:00',
                '1:00 PM': '13:00'
            }
            if text in time_conversions:
                return {'new_text': time_conversions[text], 'type': 'time_12hr_to_24hr'}
        
        return None
    
    def _get_nl_transformation(self, text: str, topic: str) -> Dict[str, Any]:
        """Get NL transformation for a candidate"""
        if topic == 'PERSON':
            return {'new_text': f'a person named {text}', 'type': 'entity_descriptor'}
        elif topic == 'ACTION':
            action_alternatives = {
                'Calculate': 'Determine',
                'walks': 'moves on foot',
                'reach': 'arrive at'
            }
            if text in action_alternatives:
                return {'new_text': action_alternatives[text], 'type': 'action_alternative'}
        elif topic == 'OBJECT':
            if 'number' in text.lower():
                return {'new_text': text.replace('number', 'value'), 'type': 'object_specification'}
        
        return None
    
    def _test_numerical_robustness(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Can the model handle different numerical representations?
        Focuses on: fraction↔decimal↔percentage, word↔digit, scientific notation
        """
        relevant_variations = []
        
        # From math domain: numerical format changes
        for var in domain_variations.get('math', []):
            if any(keyword in var.get('transformation_type', '') for keyword in 
                   ['fraction_to_decimal', 'decimal_to_fraction', 'fraction_to_percentage', 'word_to_number']):
                var['debugging_rationale'] = "Tests if model maintains accuracy across numerical formats"
                relevant_variations.append(var)
        
        # From NL domain: number word variations  
        for var in domain_variations.get('nl', []):
            if 'QUANTITY' in var.get('transformation_type', '') or 'number' in var.get('new_component', '').lower():
                var['debugging_rationale'] = "Tests numerical recognition in natural language"
                relevant_variations.append(var)
        
        # From combination variations: those involving numerical transformations
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('fraction_to_decimal' in t.get('type', '') or 'number' in t.get('type', '') 
                   for t in transformations):
                var['debugging_rationale'] = "Tests numerical robustness with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_format_dependency(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Does changing format/presentation break model understanding?
        Focuses on: time formats, date formats, mathematical notation
        """
        relevant_variations = []
        
        # From temporal domain: format changes
        for var in domain_variations.get('temporal', []):
            if any(keyword in var.get('transformation_type', '') for keyword in
                   ['time_12hr_to_24hr', 'time_24hr_to_12hr', 'date_format']):
                var['debugging_rationale'] = "Tests sensitivity to temporal format changes"
                relevant_variations.append(var)
        
        # From math domain: notation changes
        for var in domain_variations.get('math', []):
            if any(keyword in var.get('transformation_type', '') for keyword in
                   ['algebraic_equivalent', 'trigonometric_equivalent']):
                var['debugging_rationale'] = "Tests mathematical notation format robustness"
                relevant_variations.append(var)
        
        # From combination variations: those involving format changes
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('time_' in t.get('type', '') or 'format' in t.get('type', '') 
                   for t in transformations):
                var['debugging_rationale'] = "Tests format dependency with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_context_preservation(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Does rephrasing or elaboration lose essential meaning?
        Focuses on: entity elaboration, action alternatives, descriptive changes
        """
        relevant_variations = []
        
        # From NL domain: contextual changes
        for var in domain_variations.get('nl', []):
            if any(keyword in var.get('transformation_type', '') for keyword in
                   ['entity_descriptor', 'action_alternative', 'object_specification']):
                var['debugging_rationale'] = "Tests preservation of meaning through rephrasing"
                relevant_variations.append(var)
        
        # From temporal domain: expression alternatives
        for var in domain_variations.get('temporal', []):
            if any(keyword in var.get('transformation_type', '') for keyword in
                   ['relative_time', 'temporal_preposition']):
                var['debugging_rationale'] = "Tests temporal context understanding"
                relevant_variations.append(var)
        
        # From combination variations: those involving context changes
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('entity_' in t.get('type', '') or 'action_' in t.get('type', '') 
                   for t in transformations):
                var['debugging_rationale'] = "Tests context preservation with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_unit_sensitivity(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Does the model track units correctly through transformations?
        Focuses on: unit conversions that affect calculations
        """
        relevant_variations = []
        
        # From temporal domain: unit conversions
        for var in domain_variations.get('temporal', []):
            if 'duration_to_' in var.get('transformation_type', ''):
                var['debugging_rationale'] = "Tests unit tracking in temporal calculations"
                relevant_variations.append(var)
        
        # From math domain: geometric units
        for var in domain_variations.get('math', []):
            if 'geometry' in var.get('transformation_type', '') and any(unit in var.get('modified_problem', '').lower() 
                for unit in ['cm', 'mm', 'meter', 'inch', 'foot']):
                var['debugging_rationale'] = "Tests geometric unit handling"
                relevant_variations.append(var)
        
        # From combination variations: those involving unit changes
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('duration_to_' in t.get('type', '') or 'geometry_' in t.get('type', '') 
                   for t in transformations):
                var['debugging_rationale'] = "Tests unit sensitivity with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_linguistic_flexibility(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Can the model handle different ways of expressing the same concept?
        Focuses on: synonyms, elaborations, alternative phrasings
        """
        relevant_variations = []
        
        # From NL domain: linguistic alternatives
        for var in domain_variations.get('nl', []):
            var['debugging_rationale'] = "Tests linguistic robustness and synonym handling"
            relevant_variations.append(var)
        
        # From combination variations: those involving linguistic changes
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('nl' in t.get('domain', '') for t in transformations):
                var['debugging_rationale'] = "Tests linguistic flexibility with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_temporal_reasoning(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Does the model correctly handle temporal logic and relationships?
        Focuses on: time calculations, sequence understanding, duration logic
        """
        relevant_variations = []
        
        # All temporal variations test temporal reasoning
        for var in domain_variations.get('temporal', []):
            var['debugging_rationale'] = "Tests temporal logic and time-based reasoning"
            relevant_variations.append(var)
        
        # From combination variations: those involving temporal reasoning
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('temporal' in t.get('domain', '') for t in transformations):
                var['debugging_rationale'] = "Tests temporal reasoning with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_mathematical_equivalence(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Does the model understand mathematical equivalences and transformations?
        Focuses on: algebraic equivalents, geometric terminology, expression forms
        """
        relevant_variations = []
        
        # All math variations test mathematical reasoning
        for var in domain_variations.get('math', []):
            var['debugging_rationale'] = "Tests mathematical reasoning and equivalence understanding"
            relevant_variations.append(var)
        
        # From combination variations: those involving mathematical equivalence
        for var in domain_variations.get('combinations', []):
            transformations = var.get('transformations', [])
            if any('math' in t.get('domain', '') for t in transformations):
                var['debugging_rationale'] = "Tests mathematical equivalence with multiple simultaneous changes"
                relevant_variations.append(var)
        
        return relevant_variations
    
    def _test_integration_capability(self, problem: str, domain_variations: Dict) -> List[Dict[str, Any]]:
        """
        Test: Can the model handle changes across multiple domains simultaneously?
        Focuses on: cross-domain combinations, complex multi-faceted changes
        """
        relevant_variations = []
        
        # Look for existing combination variations from traditional logic
        for domain_name, variations in domain_variations.items():
            if domain_name != 'combinations':  # Handle combinations separately
                for var in variations:
                    if var.get('combination_order', 1) > 1:  # Multi-component variations
                        var['debugging_rationale'] = "Tests integration of multiple cognitive capabilities"
                        relevant_variations.append(var)
        
        # All combination variations test integration capability by definition
        for var in domain_variations.get('combinations', []):
            var['debugging_rationale'] = "Tests integration of multiple cognitive capabilities simultaneously"
            relevant_variations.append(var)
        
        return relevant_variations
    
    def _get_capability_description(self, capability: str) -> str:
        """Get human-readable description of what each capability tests"""
        descriptions = {
            'numerical_robustness': 'Tests handling of different numerical formats (fractions, decimals, percentages)',
            'format_dependency': 'Tests sensitivity to presentation format changes (time, date, notation)',
            'context_preservation': 'Tests meaning preservation through rephrasing and elaboration',
            'unit_sensitivity': 'Tests correct tracking of units through conversions and calculations',
            'linguistic_flexibility': 'Tests handling of synonyms, alternatives, and linguistic variations',
            'temporal_reasoning': 'Tests time-based logic, sequences, and temporal relationships',
            'mathematical_equivalence': 'Tests understanding of mathematical equivalences and transformations',
            'integration_capability': 'Tests multi-domain reasoning and cross-domain integration'
        }
        return descriptions.get(capability, 'Tests specific cognitive capability')
    
    def _optimize_and_prioritize_variations(self, problem: str, all_variations: List[Dict[str, Any]], 
                                           max_variations: int = None, enable_priority_scoring: bool = False) -> List[Dict[str, Any]]:
        """
        PLACEHOLDER: Optimize variations and select highest priority ones for the given problem context.
        
        FUTURE IMPLEMENTATION GOALS:
        
        1. VARIATION OPTIMIZER:
           - Learn from successful transformations across different problem types
           - Update transformation documents based on effectiveness metrics
           - Identify gaps in coverage and suggest new transformation patterns
           - Track which transformations lead to most meaningful model testing
        
        2. PRIORITY-BASED SELECTION:
           - Analyze problem context to determine most relevant transformation types
           - For geometry problems: prioritize spatial/dimensional transformations
           - For temporal problems: prioritize time/duration/sequence transformations  
           - For word problems: prioritize entity/action/quantity transformations
           - Score variations based on contextual relevance
        
        3. INTELLIGENT FILTERING:
           - Remove redundant variations that test same capability
           - Ensure diverse coverage across transformation dimensions
           - Balance near vs far transfer based on evaluation goals
           - Maintain minimum coverage requirements per domain
        
        PRIORITY SCORING CRITERIA (to be implemented):
        - Problem domain relevance (math/temporal/NL focus)
        - Transformation complexity level
        - Candidate impact on solution path
        - Coverage of untested transformation types
        - Diversity in linguistic/mathematical representation
        """
        
        # PLACEHOLDER IMPLEMENTATION: Priority scoring controlled by switch
        # TODO: Implement sophisticated priority scoring and selection
        
        if enable_priority_scoring:
            logger.debug("    🎯 Priority scoring ENABLED - calculating variation priorities...")
            # Add priority scores
            for variation in all_variations:
                variation['priority_score'] = self._calculate_variation_priority(problem, variation)
                variation['optimization_status'] = 'priority_scoring_applied'
            
            # Sort by priority (highest first)
            sorted_variations = sorted(all_variations, key=lambda x: x.get('priority_score', 0), reverse=True)
        else:
            logger.debug("    ⚠️ Priority scoring DISABLED - using original order...")
            # No priority scoring - keep original order, add placeholder metadata
            for variation in all_variations:
                variation['priority_score'] = None  # Explicitly disabled
                variation['optimization_status'] = 'priority_scoring_disabled'
            
            sorted_variations = all_variations  # Keep original order
        
        # Apply max_variations limit if specified
        if max_variations and len(sorted_variations) > max_variations:
            logger.debug(f"    🎯 Limiting to top {max_variations} highest-priority variations")
            selected_variations = sorted_variations[:max_variations]
        else:
            selected_variations = sorted_variations
        
        # Add selection metadata
        for i, variation in enumerate(selected_variations):
            variation['selection_rank'] = i + 1
            variation['total_candidates'] = len(all_variations)
        
        return selected_variations
    
    def _calculate_variation_priority(self, problem: str, variation: Dict[str, Any]) -> float:
        """
        PLACEHOLDER: Calculate priority score for a variation based on problem context.
        
        FUTURE SCORING ALGORITHM:
        1. Domain Relevance (0-30 points):
           - How well does transformation type match problem domain?
           - Geometry problems get higher scores for spatial transformations
           - Temporal problems get higher scores for time transformations
        
        2. Transformation Impact (0-25 points):
           - Does the transformation affect solution-critical components?
           - Changes to key numbers/units score higher than descriptive changes
        
        3. Linguistic Complexity (0-20 points):
           - Far transfer variations score higher than near transfer
           - Sophisticated language transformations score higher
        
        4. Coverage Diversity (0-15 points):
           - Transformations of underrepresented types score higher
           - Ensures balanced testing across transformation dimensions
        
        5. Novelty Factor (0-10 points):
           - Less common transformation patterns score higher
           - Encourages exploration of edge cases
        """
        
        # PLACEHOLDER: Simple scoring based on available metadata
        score = 50.0  # Base score
        
        # Domain matching bonus
        problem_lower = problem.lower()
        transform_type = variation.get('transformation_type', '')
        
        if 'geometry' in transform_type and any(word in problem_lower for word in ['area', 'rectangle', 'circle', 'triangle']):
            score += 15.0
        elif 'temporal' in transform_type and any(word in problem_lower for word in ['hour', 'time', 'am', 'pm', 'minute']):
            score += 15.0
        elif 'combination' in transform_type:
            score += 10.0  # Combination variations are generally valuable
        
        # Transfer type bonus
        transfer_type = variation.get('transfer_type', '')
        if transfer_type == 'far':
            score += 10.0
        elif transfer_type == 'near':
            score += 5.0
        
        # Confidence adjustment
        confidence = variation.get('confidence', '')
        if confidence == 'model_generated':
            score += 5.0
        
        return score
    
    def update_transformation_docs(self, successful_variations: List[Dict[str, Any]], 
                                 problem_types: List[str], effectiveness_metrics: Dict[str, float]):
        """
        PLACEHOLDER: Update transformation documents based on successful variation patterns.
        
        FUTURE IMPLEMENTATION:
        
        1. LEARNING FROM SUCCESS:
           - Track which transformations led to meaningful model differences
           - Identify patterns in successful variations across problem types
           - Learn optimal transformation complexity for different domains
        
        2. DOCUMENT OPTIMIZATION:
           - Add new transformation patterns that showed high effectiveness
           - Remove or deprioritize patterns that rarely provided value
           - Update transformation examples with better ones from real usage
        
        3. ADAPTIVE IMPROVEMENT:
           - Adjust transformation probabilities based on success rates
           - Develop domain-specific transformation preferences
           - Create feedback loop for continuous improvement
        
        INPUT PARAMETERS:
        - successful_variations: List of variations that led to meaningful model testing
        - problem_types: Types of problems where variations were most effective
        - effectiveness_metrics: Quantified measures of transformation value
        
        OUTPUT:
        - Updated transformation documents
        - Improved priority scoring algorithms
        - Enhanced domain-specific transformation strategies
        """
        
        # PLACEHOLDER: Log the request for future implementation
        logger.debug(f"📝 PLACEHOLDER: Optimization requested for {len(successful_variations)} successful variations")
        logger.debug(f"📊 PLACEHOLDER: Effectiveness metrics: {effectiveness_metrics}")
        logger.debug(f"🎯 PLACEHOLDER: Problem types: {problem_types}")
        logger.debug("🔄 TODO: Implement learning-based document optimization")
    
    def generate_near_transfer_variations(self, problem: str, all_candidates: List[Dict[str, Any]], 
                                        min_count: int, max_count: int) -> List[Dict[str, Any]]:
        """
        Generate NEAR TRANSFER variations: Simple, surface-level changes that are easily recognizable.
        - Direct synonyms and simple descriptive alternatives
        - Basic unit conversions (hours→minutes, cm→mm)  
        - Number-to-word transformations
        - Simple rephrasing that maintains obvious meaning
        """
        variations = []
        
        if not self.model_client:
            return variations
            
        candidate_list = [f"'{c['text']}' ({c['domain']}:{c['topic']})" for c in all_candidates[:8]]
        
        system_prompt = """You are an expert at generating NEAR TRANSFER variations - simple, surface-level changes that preserve meaning obviously.

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

        user_prompt = f"""Original problem: {problem}

Available candidates: {', '.join(candidate_list)}

Generate exactly {min_count} to {max_count} NEAR TRANSFER variations (simple, surface-level changes).

OUTPUT: Each variation on a separate line. NO headers, NO numbers, NO formatting - just the questions:"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=300, temperature=0.3
                )
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(user_prompt, system_prompt))
            
            # Parse response lines
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip()]
            
            for i, line in enumerate(lines[:max_count]):
                cleaned_line = clean_model_response(line)
                if cleaned_line and cleaned_line != problem and is_valid_question(cleaned_line):
                    variations.append({
                        'modified_problem': cleaned_line,
                        'transformation_type': 'near_transfer',
                        'original_component': 'surface_level_candidates',
                        'new_component': 'simple_alternatives',
                        'confidence': 'model_generated',
                        'transfer_type': 'near'
                    })
        
        except Exception as e:
            logger.debug(f"    Near transfer generation failed: {e}")
        
        return variations[:max_count]  # Ensure we don't exceed max
    
    def generate_far_transfer_variations(self, problem: str, all_candidates: List[Dict[str, Any]], 
                                       min_count: int, max_count: int) -> List[Dict[str, Any]]:
        """
        Generate FAR TRANSFER variations: Complex, twisted changes that require deeper understanding.
        - Sophisticated descriptive alternatives
        - Complex unit conversions across systems  
        - Creative rephrasing with multiple transformations
        - Challenging language that tests deeper comprehension
        """
        variations = []
        
        if not self.model_client:
            return variations
            
        candidate_list = [f"'{c['text']}' ({c['domain']}:{c['topic']})" for c in all_candidates[:8]]
        
        system_prompt = """You are an expert at generating FAR TRANSFER variations - complex, sophisticated changes that test deeper understanding.

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

        user_prompt = f"""Original problem: {problem}

Available candidates: {', '.join(candidate_list)}

Generate exactly {min_count} to {max_count} FAR TRANSFER variations (complex, sophisticated changes).

OUTPUT: Each variation on a separate line. NO headers, NO numbers, NO formatting - just the questions:"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=400, temperature=0.6
                )
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(user_prompt, system_prompt))
            
            # Parse response lines
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip()]
            
            for i, line in enumerate(lines[:max_count]):
                cleaned_line = clean_model_response(line)
                if cleaned_line and cleaned_line != problem and is_valid_question(cleaned_line):
                    variations.append({
                        'modified_problem': cleaned_line,
                        'transformation_type': 'far_transfer',
                        'original_component': 'deep_level_candidates',
                        'new_component': 'sophisticated_alternatives',
                        'confidence': 'model_generated',
                        'transfer_type': 'far'
                    })
        
        except Exception as e:
            logger.debug(f"    Far transfer generation failed: {e}")
        
        return variations[:max_count]  # Ensure we don't exceed max
    
    def validate_transformation_equivalence(self, original_component: str, new_component: str, domain: str) -> bool:
        """
        Validate that a transformation preserves mathematical/semantic equivalence.
        Returns True if transformation is safe, False if potentially problematic.
        """
        
        # GEOMETRIC VALIDATION
        if domain == 'math' or 'geometry' in domain:
            # Check for problematic geometric transformations
            geometry_issues = {
                ('rectangle', 'quadrilateral'): False,  # rectangle ⊂ quadrilateral (not equivalent)
                ('square', 'rectangle'): False,         # square ⊂ rectangle (not equivalent)
                ('area', 'surface area'): False,        # 2D vs 3D (not equivalent)
                ('perimeter', 'circumference'): False,  # polygon vs circle (not equivalent)
                ('triangle', 'polygon'): False,         # triangle ⊂ polygon (not equivalent)
            }
            
            # Check both directions
            key1 = (original_component.lower(), new_component.lower())
            key2 = (new_component.lower(), original_component.lower())
            
            if key1 in geometry_issues or key2 in geometry_issues:
                logger.debug(f"    ⚠️ Rejecting unsafe geometric transformation: '{original_component}' → '{new_component}'")
                return False
        
        # TEMPORAL VALIDATION  
        if domain == 'temporal':
            # Check for problematic time transformations
            temporal_issues = {
                ('am', 'pm'): False,           # Opposite times
                ('morning', 'evening'): False, # Opposite times
                ('monday', 'friday'): False,   # Different days
            }
            
            key1 = (original_component.lower(), new_component.lower())
            key2 = (new_component.lower(), original_component.lower())
            
            if key1 in temporal_issues or key2 in temporal_issues:
                logger.debug(f"    ⚠️ Rejecting unsafe temporal transformation: '{original_component}' → '{new_component}'")
                return False
        
        # UNIT CONVERSION VALIDATION
        if any(unit in original_component.lower() or unit in new_component.lower() 
               for unit in ['cm', 'mm', 'km', 'mile', 'foot', 'inch', 'meter']):
            # Ensure unit conversions maintain dimensional consistency
            length_units = ['cm', 'mm', 'km', 'mile', 'foot', 'inch', 'meter', 'metre']
            area_units = ['cm²', 'mm²', 'km²', 'square']
            
            orig_is_length = any(unit in original_component.lower() for unit in length_units)
            new_is_length = any(unit in new_component.lower() for unit in length_units)
            orig_is_area = any(unit in original_component.lower() for unit in area_units)
            new_is_area = any(unit in new_component.lower() for unit in area_units)
            
            # Reject if mixing length and area units
            if (orig_is_length and new_is_area) or (orig_is_area and new_is_length):
                logger.debug(f"    ⚠️ Rejecting dimensional mismatch: '{original_component}' → '{new_component}'")
                return False
        
        # SEMANTIC VALIDATION - check for obvious meaning conflicts
        conflicting_pairs = [
            ('calculate', 'destroy'), ('area', 'volume'), ('width', 'height'),
            ('start', 'end'), ('before', 'after'), ('left', 'right'),
            ('increase', 'decrease'), ('add', 'subtract'), ('more', 'less')
        ]
        
        for word1, word2 in conflicting_pairs:
            if ((word1 in original_component.lower() and word2 in new_component.lower()) or
                (word2 in original_component.lower() and word1 in new_component.lower())):
                logger.debug(f"    ⚠️ Rejecting semantic conflict: '{original_component}' → '{new_component}'")
                return False
        
        # COMPLETENESS CHECK - ensure no critical words are missing
        if len(new_component.strip()) < 2:
            logger.debug(f"    ⚠️ Rejecting incomplete transformation: '{original_component}' → '{new_component}'")
            return False
            
        # All checks passed
        return True
    
    def generate_generic_variations(self, problem: str) -> List[Dict[str, Any]]:
        """
        Generate generic domain-invariant variation types using model creativity.
        """
        variations = []
        
        if not self.model_client:
            return variations
        
        # Define generic variation types
        generic_types = [
            {
                'type': 'counterfactual',
                'description': 'Create a hypothetical scenario that explores what would happen if conditions were different',
                'example': 'What if the rectangle was a square instead?'
            },
            {
                'type': 'interrogative', 
                'description': 'Transform statements into questions or change question types',
                'example': 'Change from "Calculate..." to "What is the area when..."'
            },
            {
                'type': 'logical_formulation',
                'description': 'Express the problem using logical operators and formal reasoning',
                'example': 'If P then Q, given that P is true, what is Q?'
            },
            {
                'type': 'symbolic_representation',
                'description': 'Use mathematical symbols and formal notation',
                'example': 'Express using variables: A = l × w where l=15, w=20'
            },
            {
                'type': 'narrative_style',
                'description': 'Transform into a story-like narrative with characters and context',
                'example': 'Sarah is designing a garden plot shaped like a rectangle...'
            },
            {
                'type': 'irrelevant_context',
                'description': 'Add extra information that doesn\'t affect the solution',
                'example': 'On a sunny Tuesday, calculate the area of a blue rectangle...'
            }
        ]
        
        for var_type in generic_types:
            try:
                system_prompt = f"""You are an expert at generating equivalent problem variations that preserve the exact same answer and mathematical requirements.

VARIATION TYPE: {var_type['type']}
DESCRIPTION: {var_type['description']}
EXAMPLE: {var_type['example']}

CRITICAL CONSTRAINTS:
1. MUST preserve the exact same numerical answer
2. MUST maintain the same solution method and complexity
3. MUST only change presentation style, not mathematical content
4. MUST preserve question intent exactly - do NOT change what is being asked
5. MUST keep the same problem type and difficulty level
6. Generate exactly ONE variation

OUTPUT FORMAT: Respond with ONLY the varied problem text in PLAIN TEXT, no additional explanation, no markdown formatting (**bold** or *italic*), no special symbols."""

                user_prompt = f"""Transform this problem using the {var_type['type']} approach:

{problem}

Generate one {var_type['type']} variation:"""

                try:
                    if hasattr(self.model_client, 'get_model_response'):
                        responses = self.model_client.get_model_response(
                            [system_prompt], [user_prompt], 
                            max_new_tokens=200, temperature=0.7
                        )
                        response = responses[0] if responses else ""
                    else:
                        responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                        response = responses[0] if responses else ""
                    
                    if response and clean_model_response(response) != problem:
                        variations.append({
                            'modified_problem': clean_model_response(response),
                            'transformation_type': f'generic_{var_type["type"]}',
                            'original_component': 'entire_problem',
                            'new_component': 'reformulated_problem',
                            'confidence': 'model_generated'
                        })
                
                except Exception as e:
                    logger.debug(f"    Failed to generate {var_type['type']} variation: {e}")
                    
            except Exception as e:
                logger.debug(f"    Error in generic variation {var_type['type']}: {e}")
        
        return variations
    
    def generate_model_guided_combinations(self, problem: str, existing_variations: List[Dict[str, Any]], 
                                         min_needed: int) -> List[Dict[str, Any]]:
        """
        Generate model-guided combination variations that creatively combine multiple components.
        """
        variations = []
        
        if not self.model_client or min_needed <= 0:
            return variations
        
        # Extract all detected components from the problem
        all_components = []
        
        # Get components from existing variations
        for var in existing_variations:
            orig_comp = var.get('original_component', '')
            if orig_comp and orig_comp not in all_components:
                all_components.append(orig_comp)
        
        # If we have multiple components, ask model to creatively combine them
        if len(all_components) >= 2:
            system_prompt = """You are an expert at creating equivalent problem variations by creatively transforming multiple components simultaneously.

CRITICAL REQUIREMENTS:
1. MUST preserve the exact same numerical answer
2. MUST maintain mathematical validity
3. Transform multiple components together in creative ways
4. Keep the problem intent and complexity identical
5. MUST preserve question intent exactly - do NOT change what is being asked
6. MUST keep the same problem type and difficulty level
7. Generate exactly the number of variations requested

CREATIVE COMBINATION APPROACHES:
- Unit conversions across multiple measurements
- Temporal shifts affecting multiple time elements  
- Coordinate transformations of related quantities
- Perspective changes that affect multiple elements
- Alternative representations using different notation

OUTPUT FORMAT: 
- Each variation on a separate line
- No additional text or explanations
- Just the modified problem text"""

            user_prompt = f"""Original problem: {problem}

Available components to combine: {', '.join(all_components[:8])}

Create {min_needed} creative combination variations that transform multiple components together:"""

            try:
                if hasattr(self.model_client, 'get_model_response'):
                    responses = self.model_client.get_model_response(
                        [system_prompt], [user_prompt], 
                        max_new_tokens=500, temperature=0.8
                    )
                    response = responses[0] if responses else ""
                else:
                    responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                    response = responses[0] if responses else ""
                
                # Parse line-separated response
                lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip()]
                
                for i, line in enumerate(lines[:min_needed]):
                    cleaned_line = clean_model_response(line)
                    if cleaned_line and cleaned_line != problem and is_valid_question(cleaned_line):
                        variations.append({
                            'modified_problem': cleaned_line,
                            'transformation_type': 'model_guided_combination',
                            'original_component': 'multiple_components',
                            'new_component': 'creatively_combined',
                            'combination_order': len(all_components),
                            'confidence': 'model_generated'
                        })
                        
            except Exception as e:
                logger.debug(f"    Model-guided combination generation failed: {e}")
        
        return variations
    
    def generate_model_guided_variations(self, problem: str, math_topics: Dict[str, List[Tuple[str, int]]], 
                                       temporal_topics: Dict[str, List[Tuple[str, int]]]) -> List[Dict[str, Any]]:
        """
        Generate variations using model for unmapped patterns following systematic strategy:
        1. Unitary transformations (single component changes)
        2. Combination transformations (multi-component changes)  
        3. General variations (counterfactual, narrative, etc.)
        """
        variations = []
        
        if not self.model_client:
            return variations
        
        # Check if this problem has unmapped patterns (patterns not handled by deterministic engines)
        has_unmapped_patterns = False
        unmapped_components = []
        
        # Collect unmapped math components
        for topic, matches in math_topics.items():
            for match_text, match_pos in matches:
                handled = any(var['original_component'] == match_text 
                            for var in self.math_engine.apply_math_transformations(problem, {topic: [(match_text, match_pos)]}))
                if not handled:
                    has_unmapped_patterns = True
                    unmapped_components.append(f"{match_text} (math:{topic})")
        
        # Collect unmapped temporal components  
        for topic, matches in temporal_topics.items():
            for match_text, match_pos in matches:
                handled = any(var['original_component'] == match_text 
                            for var in self.temporal_engine.apply_temporal_transformations(problem, {topic: [(match_text, match_pos)]}))
                if not handled:
                    has_unmapped_patterns = True
                    unmapped_components.append(f"{match_text} (temporal:{topic})")
        
        # If we have unmapped patterns, follow systematic model-guided strategy
        if has_unmapped_patterns:
            logger.debug(f"    Found unmapped components: {unmapped_components}")
            
            # STEP 1: Model-guided unitary transformations
            logger.debug("    Generating model-guided unitary transformations...")
            unitary_variations = self._generate_model_unitary_variations(problem, unmapped_components)
            variations.extend(unitary_variations)
            logger.debug(f"      Generated {len(unitary_variations)} unitary variations")
            
            # STEP 2: Model-guided combination transformations
            logger.debug("    Generating model-guided combination transformations...")
            combination_variations = self._generate_model_combination_variations(problem, unmapped_components)
            variations.extend(combination_variations)
            logger.debug(f"      Generated {len(combination_variations)} combination variations")
            
            # STEP 3: Model-guided general variations
            logger.debug("    Generating model-guided general variations...")
            general_variations = self._generate_model_general_variations(problem)
            variations.extend(general_variations)
            logger.debug(f"      Generated {len(general_variations)} general variations")
        
        return variations
    
    def generate_comprehensive_model_variations(self, problem: str, all_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive model-guided variations for ALL detected candidates:
        - 5 Unitary: Individual candidate synonyms/replacements
        - 5 Combinations: Grouped candidate transformations  
        - 5 General: Overall problem reformulations
        """
        variations = []
        
        if not self.model_client or not all_candidates:
            return variations
        
        logger.debug(f"    Processing {len(all_candidates)} candidates from all domains...")
        
        # 1. UNITARY VARIATIONS (5): Individual candidate synonyms
        logger.debug("    Generating 5 unitary variations...")
        unitary_variations = self._generate_model_unitary_comprehensive(problem, all_candidates)
        variations.extend(unitary_variations)
        logger.debug(f"      Generated {len(unitary_variations)} unitary variations")
        
        # 2. COMBINATION VARIATIONS (5): Grouped candidate transformations
        logger.debug("    Generating 5 combination variations...")
        combination_variations = self._generate_model_combination_comprehensive(problem, all_candidates)
        variations.extend(combination_variations)
        logger.debug(f"      Generated {len(combination_variations)} combination variations")
        
        # 3. GENERAL VARIATIONS (5): Overall problem reformulations
        logger.debug("    Generating 5 general variations...")
        general_variations = self._generate_model_general_comprehensive(problem, all_candidates)
        variations.extend(general_variations)
        logger.debug(f"      Generated {len(general_variations)} general variations")
        
        return variations
    
    def _generate_model_unitary_comprehensive(self, problem: str, all_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 5 unitary variations: individual candidate synonyms that retain question intent"""
        variations = []
        
        candidate_list = [f"'{c['text']}' ({c['domain']}:{c['topic']})" for c in all_candidates[:10]]
        
        system_prompt = """You are an expert at generating sophisticated, creative transformations that preserve EXACT question intent and answer.

STEP 1: IDENTIFY IMPACTFUL CANDIDATES
- First identify which candidates actually impact the problem solution
- Focus ONLY on components that are crucial for getting the correct answer
- Skip trivial elements that don't affect the mathematical/temporal calculation

STEP 2: CREATIVE TRANSFORMATION STRATEGY
- Generate complex, sophisticated synonyms (not simple word swaps)
- For TIME: "5pm" → "the time when the clock reads five o'clock in the evening"
- For DURATION: "quarterly" → "every three months" 
- For MATH: "3/4" → "three quarters of the total amount"
- For ENTITIES: "rectangle" → "a four-sided geometric figure with right angles"
- Make replacements challenging yet meaningful - avoid simple substitutions

STEP 3: COMPOUND CANDIDATE TRANSFORMATION
For candidates with number+unit combinations, consider multiple transformation approaches:

A) NUMBER TRANSFORMATIONS:
- "9 AM" → "nine o'clock in the morning" (number to word)
- "20 eggs" → "twenty chicken eggs" (number to word + descriptor)

B) UNIT TRANSFORMATIONS:
- "9 AM" → "9 o'clock in the morning" (unit descriptor change)
- "20 eggs" → "20 pieces of eggs" (unit alternative)

C) CONVERSION TRANSFORMATIONS (MUST BE MATHEMATICALLY EXACT):
- "9 AM" → "21:00 hours" (12hr to 24hr: 9 + 12 = 21) ✓ CORRECT
- "20 eggs" → "1.67 dozen eggs" (20 ÷ 12 = 1.67) ✓ CORRECT
- "5 miles" → "8.047 kilometers" (5 × 1.609 = 8.047) ✓ CORRECT
- "2 hours" → "120 minutes" (2 × 60 = 120) ✓ CORRECT
- "30 minutes" → "half an hour" (30 = 60 × 0.5) ✓ CORRECT

CRITICAL: Verify ALL conversion math before applying:
✗ WRONG: "30 minutes" → "three quarters of an hour" (30 ≠ 45, this is INCORRECT!)
✗ WRONG: "45 minutes" → "half an hour" (45 ≠ 30, this is INCORRECT!)
✗ WRONG: "1.5 hours" → "60 minutes" (90 ≠ 60, this is INCORRECT!)

D) COMBINED TRANSFORMATIONS:
- "9 AM" → "twenty-one hundred hours" (conversion + descriptive)
- "20 eggs" → "one and two-thirds dozen chicken eggs" (conversion + descriptors)

MANDATORY: Always verify conversion factors are mathematically correct to maintain identical final answers

CRITICAL CONSTRAINTS:
1. Transform only ONE impactful candidate per variation
2. Use sophisticated, creative language that requires deeper understanding
3. Final answer must remain identical
4. Question intent must be preserved exactly - no alteration to the fundamental question
5. Replacements should be complex but still unambiguous

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- MUST use EXACTLY this format: candidate_text → replacement_text | complete_modified_problem
- Each variation on its own line
- PLAIN TEXT ONLY - no **, no *, no markdown formatting
- NO headers like "Variation 1" or "Examples:" or "Output:"
- NO meta-text, NO explanations
- NO numbered lists (1., 2., etc.)
- JUST the variations in the required format

EXAMPLE:
rectangle → rectangular shape | Calculate the area of a rectangular shape that is 15 cm by 20 cm.
2 hours → 120 minutes | John walks 5 miles in 120 minutes to reach the meeting at 3:00 PM."""

        user_prompt = f"""Original problem: {problem}

Available candidates: {', '.join(candidate_list)}

INSTRUCTIONS:
1. First, identify which candidates are most IMPACTFUL to the solution
2. Focus on candidates that actually affect the mathematical/temporal calculation
3. VERIFY all numerical conversions are mathematically correct before using them
4. Generate exactly 5 sophisticated unitary variations using creative, complex language

Examples of CORRECT sophisticated transformations:
   - "5pm" → "the time when the clock reads five o'clock in the evening"
   - "quarterly" → "every three months in succession"
   - "3/4" → "three quarters of the total amount"
   - "9 AM" → "twenty-one hundred hours" (9+12=21, CORRECT!)
   - "2 hours" → "one hundred and twenty minutes" (2×60=120, CORRECT!)
   - "30 minutes" → "half an hour" (30=60×0.5, CORRECT!)
   - "5 miles" → "eight point zero four seven kilometers" (5×1.609=8.047, CORRECT!)

Examples to AVOID (mathematically incorrect):
   ✗ "30 minutes" → "three quarters of an hour" (30 ≠ 45, WRONG!)
   ✗ "45 minutes" → "half an hour" (45 ≠ 30, WRONG!)

Generate exactly 5 unitary variations (one impactful candidate replacement each):"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=400, temperature=0.4
                )
                response = responses[0] if responses else ""
            else:
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            
            # Parse response
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip() and '→' in line and '|' in line]
            
            for line in lines[:5]:  # Exactly 5 variations
                try:
                    parts = line.split('|')
                    if len(parts) == 2:
                        transformation = parts[0].strip()
                        modified_problem = parts[1].strip()
                        
                        if '→' in transformation:
                            orig_comp, new_comp = [x.strip() for x in transformation.split('→')]
                            
                            variations.append({
                                'modified_problem': modified_problem,
                                'transformation_type': 'model_unitary_comprehensive',
                                'original_component': orig_comp,
                                'new_component': new_comp,
                                'confidence': 'model_generated'
                            })
                except Exception as e:
                    logger.debug(f"      Failed to parse unitary line: {line} - {e}")
        
        except Exception as e:
            logger.debug(f"    Model unitary comprehensive generation failed: {e}")
        
        return variations
    
    def _generate_model_combination_comprehensive(self, problem: str, all_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 5 combination variations: grouped candidate transformations that work together"""
        variations = []
        
        candidate_list = [f"'{c['text']}' ({c['domain']}:{c['topic']})" for c in all_candidates[:10]]
        
        system_prompt = """You are an expert at generating sophisticated grouped transformations that preserve EXACT question intent and answer.

STEP 1: IDENTIFY IMPACTFUL CANDIDATE GROUPS
- First identify which candidates actually impact the problem solution
- Focus on components that work together in the calculation
- Group related elements that can be transformed cohesively

STEP 2: SOPHISTICATED COMBINATION STRATEGY
- Transform MULTIPLE candidates with creative, complex language
- TIME COMBINATIONS: "6:30 AM" + "2 hours" → "half past six in the morning" + "a duration of one hundred and twenty minutes"
- MATH COMBINATIONS: "15 cm" + "area" → "fifteen centimeters in length" + "the total surface covered"
- MIXED DOMAIN: "3 hours" + "5 miles" → "three full hours of time" + "five miles of distance traveled"
- Create sophisticated transformations that require deeper understanding

STEP 3: COMPOUND CANDIDATE COMBINATIONS
For multiple candidates with number+unit combinations, use coordinated transformations:

A) COORDINATED NUMBER CHANGES:
- ["9 AM", "20 eggs"] → ["nine in the morning", "twenty chicken eggs"]

B) COORDINATED UNIT CHANGES:
- ["6:30 AM", "2 hours"] → ["six-thirty in the morning", "two full hours"]

C) COORDINATED CONVERSIONS (MUST BE MATHEMATICALLY EXACT):
- ["9 AM", "5 miles"] → ["21:00 hours", "8.047 kilometers"] (9+12=21, 5×1.609=8.047) ✓ CORRECT
- ["20 eggs", "3 hours"] → ["1.67 dozen eggs", "180 minutes"] (20÷12=1.67, 3×60=180) ✓ CORRECT
- ["30 minutes", "2:00 PM"] → ["half an hour", "14:00"] (30=60×0.5, 2+12=14) ✓ CORRECT

CRITICAL: Verify ALL conversion math before applying:
✗ WRONG: ["30 minutes", "store"] → ["three quarters of an hour", "shop"] (30 ≠ 45, INCORRECT!)
✗ WRONG: ["45 minutes", "3 PM"] → ["half an hour", "15:00"] (45 ≠ 30, INCORRECT!)
✗ WRONG: ["1.5 hours", "walk"] → ["60 minutes", "stroll"] (90 ≠ 60, INCORRECT!)

D) MIXED TRANSFORMATIONS:
- ["9 AM", "20 eggs"] → ["twenty-one hundred hours", "twenty chicken eggs"]
- ["5 miles", "2 hours"] → ["eight point zero four seven kilometers", "one hundred and twenty minutes"]

MANDATORY: Always ensure all conversion factors are correctly applied across all candidates to maintain identical answer

CRITICAL CONSTRAINTS:
1. Transform MULTIPLE impactful candidates per variation in a coherent way
2. Use complex, creative language for all transformations in the group
3. Maintain logical relationships between transformed elements
4. Final answer must remain identical
5. Question intent must be preserved exactly - no alteration to the fundamental question
6. Avoid simple word swaps - make each transformation sophisticated

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- MUST use EXACTLY this format: [candidate1→replacement1, candidate2→replacement2] | complete_modified_problem
- Each variation on its own line
- PLAIN TEXT ONLY - no **, no *, no markdown formatting
- NO headers like "Variation 1" or "Examples:" or "Output:"
- NO meta-text, NO explanations
- NO numbered lists (1., 2., etc.)
- JUST the variations in the required format

EXAMPLE:
[2 hours→120 minutes, 5 miles→8047 meters] | John walks 8047 meters in 120 minutes to reach the meeting at 3:00 PM.
[rectangle→rectangular shape, area→total area] | Calculate the total area of a rectangular shape that is 15 cm by 20 cm."""

        user_prompt = f"""Original problem: {problem}

Available candidates: {', '.join(candidate_list)}

INSTRUCTIONS:
1. First, identify which candidates are most IMPACTFUL to the solution
2. Group related candidates that work together in the calculation
3. VERIFY all numerical conversions are mathematically correct before using them
4. Generate exactly 5 sophisticated combination variations using creative, complex language

Examples of CORRECT sophisticated grouped transformations:
   - ["6:30 AM", "2 hours"] → ["half past six in the morning", "one hundred and twenty minutes"] (2×60=120, CORRECT!)
   - ["15 cm", "area"] → ["fifteen centimeters in length", "the total surface covered"]
   - ["9 AM", "5 miles"] → ["twenty-one hundred hours", "eight point zero four seven kilometers"] (9+12=21, 5×1.609=8.047, CORRECT!)
   - ["30 minutes", "2:00 PM"] → ["half an hour", "fourteen hundred hours"] (30=60×0.5, 2+12=14, CORRECT!)

Examples to AVOID (mathematically incorrect):
   ✗ ["30 minutes", "store"] → ["three quarters of an hour", "shop"] (30 ≠ 45, WRONG!)
   ✗ ["45 minutes", "3 PM"] → ["half an hour", "15:00"] (45 ≠ 30, WRONG!)

Generate exactly 5 combination variations (multiple impactful candidate transformations each):"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=500, temperature=0.5
                )
                response = responses[0] if responses else ""
            else:
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            
            # Parse response
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip() and '|' in line and '[' in line]
            
            for line in lines[:5]:  # Exactly 5 variations
                try:
                    parts = line.split('|')
                    if len(parts) == 2:
                        transformations = parts[0].strip()
                        modified_problem = parts[1].strip()
                        
                        # Extract multiple transformations
                        if transformations.startswith('[') and transformations.endswith(']'):
                            trans_content = transformations[1:-1]
                            individual_trans = [t.strip() for t in trans_content.split(',')]
                            
                            orig_components = []
                            new_components = []
                            
                            for trans in individual_trans:
                                if '→' in trans:
                                    orig, new = [x.strip() for x in trans.split('→')]
                                    orig_components.append(orig)
                                    new_components.append(new)
                            
                            if orig_components and new_components:
                                variations.append({
                                    'modified_problem': modified_problem,
                                    'transformation_type': 'model_combination_comprehensive',
                                    'original_component': '; '.join(orig_components),
                                    'new_component': '; '.join(new_components),
                                    'combination_order': len(orig_components),
                                    'confidence': 'model_generated'
                                })
                
                except Exception as e:
                    logger.debug(f"      Failed to parse combination line: {line} - {e}")
        
        except Exception as e:
            logger.debug(f"    Model combination comprehensive generation failed: {e}")
        
        return variations
    
    def _generate_model_general_comprehensive(self, problem: str, all_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 5 general variations: overall problem reformulations maintaining intent"""
        variations = []
        
        candidate_list = [f"'{c['text']}' ({c['domain']}:{c['topic']})" for c in all_candidates[:10]]
        
        # Define 5 specific general variation types
        general_types = [
            {
                'type': 'narrative_contextual',
                'description': 'Transform into a story-like narrative with real-world context while preserving all mathematical relationships'
            },
            {
                'type': 'formal_mathematical', 
                'description': 'Express using formal mathematical notation and variables while maintaining exact calculations'
            },
            {
                'type': 'interrogative_alternative',
                'description': 'Restructure the question format or questioning approach while seeking the same answer'
            },
            {
                'type': 'conditional_hypothetical',
                'description': 'Frame as conditional or hypothetical scenario while preserving core mathematical relationships'
            },
            {
                'type': 'procedural_stepwise',
                'description': 'Reframe as step-by-step procedure or process while maintaining the same solution path'
            }
        ]
        
        for var_type in general_types:
            try:
                system_prompt = f"""You are an expert at generating sophisticated problem reformulations that preserve EXACT answer and intent.

GENERAL TRANSFORMATION TYPE: {var_type['type']}
STRATEGY: {var_type['description']}

SOPHISTICATED TRANSFORMATION EXAMPLES:
- TIME: "5pm" → "the time when the clock reads five o'clock in the evening"
- DURATION: "quarterly" → "every three months in succession"
- FRACTIONS: "3/4" → "three quarters of the total amount"
- GEOMETRY: "rectangle" → "a four-sided geometric figure with parallel opposite sides"
- DISTANCE: "5 miles" → "five miles of ground covered"
- Use complex, descriptive language that requires deeper comprehension

CRITICAL CONSTRAINTS:
1. MUST preserve the exact same numerical answer
2. MUST maintain the same solution method and complexity
3. MUST keep all mathematical relationships intact
4. MUST preserve question intent exactly - do NOT change what is being asked
5. MUST keep the same problem type and difficulty level
6. Use sophisticated, creative language throughout (avoid simple word swaps)
7. Generate exactly ONE variation of this type

OUTPUT FORMAT: Respond with ONLY the reformulated problem text, no additional explanation."""

                user_prompt = f"""Original problem: {problem}

Detected candidates: {', '.join(candidate_list)}

Transform using the {var_type['type']} approach:"""

                if hasattr(self.model_client, 'get_model_response'):
                    responses = self.model_client.get_model_response(
                        [system_prompt], [user_prompt], 
                        max_new_tokens=250, temperature=0.7
                    )
                    response = responses[0] if responses else ""
                else:
                    responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                    response = responses[0] if responses else ""
                
                cleaned_response = clean_model_response(response)
                if cleaned_response and cleaned_response != problem and is_valid_question(cleaned_response):
                    variations.append({
                        'modified_problem': cleaned_response,
                        'transformation_type': f'model_general_{var_type["type"]}',
                        'original_component': 'entire_problem_comprehensive',
                        'new_component': 'reformulated_comprehensive',
                        'confidence': 'model_generated'
                    })
            
            except Exception as e:
                logger.debug(f"      Failed to generate {var_type['type']} variation: {e}")
        
        return variations
    
    def _generate_model_unitary_variations(self, problem: str, unmapped_components: List[str]) -> List[Dict[str, Any]]:
        """Generate model-guided unitary transformations for unmapped components"""
        variations = []
        
        system_prompt = """You are an expert at generating equivalent unitary transformations that preserve the EXACT same answer and meaning.

UNITARY TRANSFORMATION STRATEGY:
- Transform ONLY ONE component at a time
- Preserve exact numerical answers and logical meaning
- Use equivalent representations (units, expressions, formulations)
- Maintain mathematical and temporal accuracy
- MUST preserve question intent exactly - do NOT change what is being asked

CRITICAL CONSTRAINTS:
1. MUST preserve the final answer/result
2. MUST change only ONE element per variation
3. MUST maintain mathematical/temporal validity
4. MUST preserve question intent exactly - no alteration to the fundamental question
5. MUST keep the same problem type and difficulty level
6. Generate exactly 2 unitary variations

OUTPUT FORMAT:
- Each variation on a separate line
- Format: original_component → new_component | modified_problem_text
- NO additional explanations or text

EXAMPLE:
2 hours → 120 minutes | John walks 5 miles in 120 minutes to reach the meeting
3/4 → 0.75 | If 0.75 of a number is 24, what is 1/2 of that number?"""

        user_prompt = f"""Problem: {problem}

Unmapped components to transform: {', '.join(unmapped_components)}

Generate exactly 2 unitary transformations (one component each):"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=300, temperature=0.4
                )
                response = responses[0] if responses else ""
            else:
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            
            # Parse response
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip() and '→' in line and '|' in line]
            
            for line in lines[:2]:  # Limit to 2 variations
                try:
                    parts = line.split('|')
                    if len(parts) == 2:
                        transformation = parts[0].strip()
                        modified_problem = parts[1].strip()
                        
                        if '→' in transformation:
                            orig_comp, new_comp = [x.strip() for x in transformation.split('→')]
                            
                            variations.append({
                                'modified_problem': modified_problem,
                                'transformation_type': 'model_guided_unitary',
                                'original_component': orig_comp,
                                'new_component': new_comp,
                                'confidence': 'model_generated'
                            })
                except Exception as e:
                    logger.debug(f"      Failed to parse unitary variation: {line} - {e}")
        
        except Exception as e:
            logger.debug(f"    Model unitary generation failed: {e}")
        
        return variations
    
    def _generate_model_combination_variations(self, problem: str, unmapped_components: List[str]) -> List[Dict[str, Any]]:
        """Generate model-guided combination transformations for unmapped components"""
        variations = []
        
        system_prompt = """You are an expert at generating equivalent combination transformations that preserve the EXACT same answer and meaning.

COMBINATION TRANSFORMATION STRATEGY:
- Transform MULTIPLE components simultaneously 
- Maintain relationships between transformed elements
- Preserve exact numerical answers and logical meaning
- Create coherent multi-component changes
- MUST preserve question intent exactly - do NOT change what is being asked

CRITICAL CONSTRAINTS:
1. MUST preserve the final answer/result
2. MUST change MULTIPLE elements per variation
3. MUST maintain consistency across changes
4. MUST preserve question intent exactly - no alteration to the fundamental question
5. MUST keep the same problem type and difficulty level
6. Generate exactly 2 combination variations

OUTPUT FORMAT:
- Each variation on a separate line
- Format: [comp1→new1, comp2→new2] | modified_problem_text
- NO additional explanations or text

EXAMPLE:
[2 hours→120 minutes, 5 miles→8047 meters] | John walks 8047 meters in 120 minutes to reach the meeting"""

        user_prompt = f"""Problem: {problem}

Unmapped components available: {', '.join(unmapped_components)}

Generate exactly 2 combination transformations (multiple components each):"""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [system_prompt], [user_prompt], 
                    max_new_tokens=400, temperature=0.5
                )
                response = responses[0] if responses else ""
            else:
                responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                response = responses[0] if responses else ""
            
            # Parse response
            lines = [line.strip() for line in clean_model_response(response).split('\n') if line.strip() and '|' in line]
            
            for line in lines[:2]:  # Limit to 2 variations
                try:
                    parts = line.split('|')
                    if len(parts) == 2:
                        transformations = parts[0].strip()
                        modified_problem = parts[1].strip()
                        
                        # Extract multiple transformations
                        if transformations.startswith('[') and transformations.endswith(']'):
                            trans_content = transformations[1:-1]
                            individual_trans = [t.strip() for t in trans_content.split(',')]
                            
                            orig_components = []
                            new_components = []
                            
                            for trans in individual_trans:
                                if '→' in trans:
                                    orig, new = [x.strip() for x in trans.split('→')]
                                    orig_components.append(orig)
                                    new_components.append(new)
                            
                            if orig_components and new_components:
                                variations.append({
                                    'modified_problem': modified_problem,
                                    'transformation_type': 'model_guided_combination',
                                    'original_component': '; '.join(orig_components),
                                    'new_component': '; '.join(new_components),
                                    'combination_order': len(orig_components),
                                    'confidence': 'model_generated'
                                })
                
                except Exception as e:
                    logger.debug(f"      Failed to parse combination variation: {line} - {e}")
        
        except Exception as e:
            logger.debug(f"    Model combination generation failed: {e}")
        
        return variations
    
    def _generate_model_general_variations(self, problem: str) -> List[Dict[str, Any]]:
        """Generate model-guided general variations (counterfactual, narrative, etc.)"""
        variations = []
        
        # Define general variation types for unmapped problems
        general_types = [
            {
                'type': 'counterfactual_unmapped',
                'description': 'Create hypothetical scenarios for unmapped problem elements',
                'instruction': 'Transform into a what-if scenario while preserving the core mathematical relationship'
            },
            {
                'type': 'narrative_unmapped', 
                'description': 'Add story context to unmapped problem elements',
                'instruction': 'Transform into a story format while maintaining exact calculations'
            },
            {
                'type': 'formal_unmapped',
                'description': 'Use formal mathematical representation for unmapped elements',
                'instruction': 'Express using formal mathematical notation and variables'
            }
        ]
        
        for var_type in general_types:
            try:
                system_prompt = f"""You are an expert at generating equivalent problem variations that preserve the exact same answer.

VARIATION TYPE: {var_type['type']}
STRATEGY: {var_type['description']}
INSTRUCTION: {var_type['instruction']}

CRITICAL CONSTRAINTS:
1. MUST preserve the exact same numerical answer
2. MUST maintain the same solution method and complexity
3. MUST only change presentation style, not mathematical content
4. MUST preserve question intent exactly - do NOT change what is being asked
5. MUST keep the same problem type and difficulty level
6. Generate exactly ONE variation

OUTPUT FORMAT: Respond with ONLY the varied problem text in PLAIN TEXT, no additional explanation, no markdown formatting (**bold** or *italic*), no special symbols."""

                user_prompt = f"""Transform this unmapped problem using the {var_type['type']} approach:

{problem}

Generate one {var_type['type']} variation:"""

                if hasattr(self.model_client, 'get_model_response'):
                    responses = self.model_client.get_model_response(
                        [system_prompt], [user_prompt], 
                        max_new_tokens=250, temperature=0.7
                    )
                    response = responses[0] if responses else ""
                else:
                    responses = self.model_client.get_model_response([system_prompt], [user_prompt])
                    response = responses[0] if responses else ""
                
                cleaned_response = clean_model_response(response)
                if cleaned_response and cleaned_response != problem and is_valid_question(cleaned_response):
                    variations.append({
                        'modified_problem': cleaned_response,
                        'transformation_type': f'model_guided_{var_type["type"]}',
                        'original_component': 'entire_problem_unmapped',
                        'new_component': 'reformulated_unmapped',
                        'confidence': 'model_generated'
                    })
            
            except Exception as e:
                logger.debug(f"      Failed to generate {var_type['type']} variation: {e}")
        
        return variations
    
    def _get_transformation_examples(self, topic: str) -> str:
        """Get example transformations for a math topic to guide the model"""
        examples = {
            'geometry': 'area → total area, perimeter → boundary length, rectangle → rectangular shape',
            'fractions': '1/2 → 0.5, 3/4 → 0.75, 2/3 → 0.667',
            'algebra': 'x + 5 → 5 + x, 2x → x * 2, x² → x^2',
            'arithmetic': '10 + 5 → 15, 4 * 3 → 12, 20 / 4 → 5',
            'trigonometry': 'sin(x) → sine(x), cos(x) → cosine(x), tan(x) → tangent(x)'
        }
        return examples.get(topic, f'{topic} expressions can be reformulated while preserving meaning')
    
    def _get_temporal_examples(self, topic: str) -> str:
        """Get example transformations for a temporal topic to guide the model"""
        examples = {
            'times_12hr': '2:30 PM → 14:30, 10:00 AM → 10:00',
            'times_24hr': '14:30 → 2:30 PM, 09:00 → 9:00 AM',
            'durations': '2 hours → 120 minutes, 30 minutes → 1800 seconds',
            'dates': 'January 15, 2023 → 2023-01-15, 01/15/2023 → 2023-01-15',
            'temporal_prep': 'before → prior to, after → following, during → throughout'
        }
        return examples.get(topic, f'{topic} expressions can be reformulated while preserving temporal meaning')
    
    def generate_combination_variations(self, problem: str, math_variations: List[Dict[str, Any]], 
                                      nl_variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate combination variations by applying multiple transformations simultaneously.
        Creates 2-way, 3-way, and higher-order combinations.
        """
        combination_variations = []
        
        # Combine all available transformations
        all_transformations = math_variations + nl_variations
        
        if len(all_transformations) < 2:
            return []  # Need at least 2 transformations to combine
        
        # Generate 2-way combinations
        for i, trans1 in enumerate(all_transformations):
            for j, trans2 in enumerate(all_transformations[i+1:], i+1):
                # Skip if transformations conflict (same component)
                if self._transformations_conflict(trans1, trans2):
                    continue
                
                # Apply both transformations
                combined_result = self._apply_multiple_transformations(problem, [trans1, trans2])
                if combined_result:
                    combination_variations.append({
                        'modified_problem': combined_result,
                        'transformation_type': 'combination_2way',
                        'combined_transformations': [
                            trans1['transformation_type'], 
                            trans2['transformation_type']
                        ],
                        'original_components': [
                            trans1.get('original_component', ''),
                            trans2.get('original_component', '')
                        ],
                        'new_components': [
                            trans1.get('new_component', ''),
                            trans2.get('new_component', '')
                        ],
                        'combination_order': 2
                    })
        
        # Generate 3-way combinations (if enough transformations available)
        if len(all_transformations) >= 3:
            # Limit 3-way combinations to prevent explosion
            max_3way = min(5, len(all_transformations))
            selected_transformations = all_transformations[:max_3way]
            
            for i, trans1 in enumerate(selected_transformations):
                for j, trans2 in enumerate(selected_transformations[i+1:], i+1):
                    for k, trans3 in enumerate(selected_transformations[j+1:], j+1):
                        # Skip if any transformations conflict
                        if (self._transformations_conflict(trans1, trans2) or 
                            self._transformations_conflict(trans1, trans3) or 
                            self._transformations_conflict(trans2, trans3)):
                            continue
                        
                        # Apply all three transformations
                        combined_result = self._apply_multiple_transformations(problem, [trans1, trans2, trans3])
                        if combined_result:
                            combination_variations.append({
                                'modified_problem': combined_result,
                                'transformation_type': 'combination_3way',
                                'combined_transformations': [
                                    trans1['transformation_type'], 
                                    trans2['transformation_type'],
                                    trans3['transformation_type']
                                ],
                                'original_components': [
                                    trans1.get('original_component', ''),
                                    trans2.get('original_component', ''),
                                    trans3.get('original_component', '')
                                ],
                                'new_components': [
                                    trans1.get('new_component', ''),
                                    trans2.get('new_component', ''),
                                    trans3.get('new_component', '')
                                ],
                                'combination_order': 3
                            })
        
        return combination_variations
    
    def _transformations_conflict(self, trans1: Dict[str, Any], trans2: Dict[str, Any]) -> bool:
        """Check if two transformations conflict (affect the same component)"""
        orig1 = trans1.get('original_component', '')
        orig2 = trans2.get('original_component', '')
        
        if not orig1 or not orig2:
            return False
        
        # Check for exact match or overlap
        return orig1 == orig2 or orig1 in orig2 or orig2 in orig1
    
    def _apply_multiple_transformations(self, problem: str, transformations: List[Dict[str, Any]]) -> str:
        """Apply multiple transformations to a problem text"""
        result = problem
        
        # Sort transformations by position (apply from end to start to preserve positions)
        sorted_transformations = []
        for trans in transformations:
            orig_comp = trans.get('original_component', '')
            if orig_comp and orig_comp in result:
                pos = result.find(orig_comp)
                if pos >= 0:
                    sorted_transformations.append((pos, trans))
        
        # Sort by position (descending to apply from end first)
        sorted_transformations.sort(key=lambda x: x[0], reverse=True)
        
        # Apply each transformation
        for pos, trans in sorted_transformations:
            orig_comp = trans.get('original_component', '')
            new_comp = trans.get('new_component', '')
            
            if orig_comp and new_comp and orig_comp in result:
                result = result.replace(orig_comp, new_comp, 1)
        
        return result if result != problem else None
    
    def _enhance_single_variation(self, original_problem: str, variation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a single variation by checking for consistency issues and related entities.
        """
        enhanced = variation.copy()
        
        if not self.consistency_enhancement_enabled:
            return enhanced
        
        try:
            # Get modified problem
            modified_problem = variation.get('modified_problem', '')
            if not modified_problem:
                return enhanced
            
            # Use existing consistency enhancement logic
            enhancer = self.get_consistency_enhancer()
            if enhancer:
                # Check for related entities that should also be changed
                original_component = variation.get('original_component', '')
                new_component = variation.get('new_component', '')
                
                if original_component and new_component:
                    # Find related entities via spaCy dependency parsing
                    related_entities = enhancer._find_related_via_spacy(original_problem, original_component)
                    
                    # Apply consistency transformations
                    final_text = modified_problem
                    for related_text, (start, end) in related_entities[:3]:  # Limit to prevent over-transformation
                        if related_text != original_component and related_text in final_text:
                            # Apply similar transformation pattern
                            if self._should_transform_related_entity(original_component, new_component, related_text):
                                transformed_related = self._transform_related_entity(original_component, new_component, related_text)
                                if transformed_related:
                                    final_text = final_text.replace(related_text, transformed_related, 1)
                    
                    enhanced['modified_problem'] = final_text
            
        except Exception as e:
            logger.debug(f"    Consistency enhancement warning: {e}")
        
        return enhanced
    
    def get_consistency_enhancer(self):
        """Get the consistency enhancer if available"""
        try:
            # Return the existing consistency enhancer
            if hasattr(self, 'consistency_enhancer'):
                return self.consistency_enhancer
            elif hasattr(self, 'variation_consistency_enhancer'):
                return self.variation_consistency_enhancer
            else:
                # Create a minimal enhancer for basic functionality
                from types import SimpleNamespace
                enhancer = SimpleNamespace()
                enhancer._find_related_via_spacy = self._find_related_via_spacy_basic
                return enhancer
        except:
            return None
    
    def _find_related_via_spacy_basic(self, problem: str, component: str) -> list:
        """Basic spaCy-based related entity detection"""
        related = []
        try:
            if self.nlp:
                doc = self.nlp(problem)
                component_doc = self.nlp(component)
                
                # Find similar dependency patterns
                for token in component_doc:
                    for other_token in doc:
                        if (other_token.dep_ == token.dep_ and 
                            other_token.pos_ == token.pos_ and 
                            other_token.text != token.text):
                            related.append((other_token.text, (other_token.idx, other_token.idx + len(other_token.text))))
        except:
            pass
        return related
    
    def _should_transform_related_entity(self, original: str, new: str, related: str) -> bool:
        """Check if a related entity should be transformed based on the transformation pattern"""
        try:
            # Simple heuristics for when to apply similar transformations
            # Number/unit transformations
            if any(char.isdigit() for char in original) and any(char.isdigit() for char in related):
                return True
            
            # Same word type transformations  
            if original.lower() in related.lower() or related.lower() in original.lower():
                return True
            
            # Temporal transformations
            temporal_words = ['hour', 'minute', 'second', 'day', 'week', 'month', 'year', 'am', 'pm']
            if any(word in original.lower() for word in temporal_words) and any(word in related.lower() for word in temporal_words):
                return True
                
        except:
            pass
        return False
    
    def _transform_related_entity(self, original: str, new: str, related: str) -> str:
        """Transform a related entity based on the transformation pattern applied to the original"""
        try:
            # Extract transformation pattern and apply to related entity
            
            # Number extraction and transformation
            import re
            orig_numbers = re.findall(r'\d+', original)
            new_numbers = re.findall(r'\d+', new)
            related_numbers = re.findall(r'\d+', related)
            
            if orig_numbers and new_numbers and related_numbers and len(orig_numbers) == len(new_numbers):
                # Apply same numerical transformation
                result = related
                for i, (orig_num, new_num, rel_num) in enumerate(zip(orig_numbers, new_numbers, related_numbers)):
                    try:
                        # Calculate transformation ratio
                        if float(orig_num) != 0:
                            ratio = float(new_num) / float(orig_num)
                            transformed_num = str(int(float(rel_num) * ratio))
                            result = result.replace(rel_num, transformed_num, 1)
                    except:
                        continue
                return result
            
            # Unit transformation
            units_mapping = {
                'cm': {'cm': 'cm', 'inches': 'inches', 'feet': 'feet'},
                'inches': {'cm': 'cm', 'inches': 'inches', 'feet': 'feet'},
                'hours': {'minutes': 'minutes', 'seconds': 'seconds', 'hours': 'hours'},
                'minutes': {'hours': 'hours', 'seconds': 'seconds', 'minutes': 'minutes'}
            }
            
            # Simple unit replacement
            for old_unit, new_unit in units_mapping.items():
                if old_unit in original.lower() and old_unit in related.lower():
                    for target_unit in new_unit:
                        if target_unit in new.lower():
                            return related.replace(old_unit, target_unit)
            
            return related
            
        except:
            return related
