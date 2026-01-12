#!/usr/bin/env python3
"""
LLM Answer Matcher - Separate Script for Answer Matching

Uses the existing LLM judge to match model answers with ground truth.
Designed to be used by the complete variation pipeline for evaluation.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports

sys.path.append(str(Path(__file__).parent))

from benchdrift.eval.llm_judge_answer_checker import LLMJudgeAnswerChecker



logger = logging.getLogger('BenchDrift')
class StringBasedMatcher:
    """Fallback string-based answer matcher when LLM is not available."""

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""

        # Convert to lowercase and strip
        answer = answer.lower().strip()

        # Remove common prefixes/suffixes
        prefixes = ['the answer is', 'answer:', 'solution:', 'result:', 'final answer:']
        suffixes = ['.', '!', '?']

        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        for suffix in suffixes:
            if answer.endswith(suffix):
                answer = answer[:-len(suffix)].strip()

        # Extract numbers if present
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[0]

        return answer

    @staticmethod
    def string_match_answers(ground_truth: str, model_answer: str) -> Dict[str, Any]:
        """Basic string matching for answers."""
        gt_norm = StringBasedMatcher.normalize_answer(ground_truth)
        ma_norm = StringBasedMatcher.normalize_answer(model_answer)

        # Exact match
        exact_match = gt_norm == ma_norm

        # Partial match (one contains the other)
        partial_match = gt_norm in ma_norm or ma_norm in gt_norm

        # Confidence based on match type
        if exact_match:
            confidence = "high"
        elif partial_match:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            'is_correct': exact_match,
            'confidence': confidence,
            'explanation': f"String match: GT='{gt_norm}', Model='{ma_norm}'",
            'judge_response': f"Fallback string matching used",
            'success': True,
            'error_message': ''
        }


class LLMAnswerMatcher:
    """Wrapper for LLM judge answer matching functionality with string fallback."""

    def __init__(self, client_type: str = 'rits', model_name: str = 'mistral_small_3_2_instruct', use_llm: bool = True, batch_size: int = 50):
        """Initialize the LLM answer matcher."""
        self.use_llm = use_llm
        self.judge = None

        if use_llm:
            try:
                self.judge = LLMJudgeAnswerChecker(
                    client_type=client_type,
                    model_name=model_name,
                    batch_size=batch_size
                )
            except Exception as e:
                logger.debug(f"⚠️ Failed to initialize LLM judge: {e}")
                logger.debug("   Falling back to string-based matching")
                self.use_llm = False

    def check_answer_match(self, ground_truth: str, model_answer: str,
                          problem_context: str = "") -> Dict[str, Any]:
        """
        Check if model answer matches ground truth using LLM judge with string fallback.

        Args:
            ground_truth: The correct answer
            model_answer: The model's answer to check
            problem_context: Optional problem context for better evaluation

        Returns:
            Dict with matching results including:
            - is_correct: bool
            - confidence: str
            - explanation: str
            - judge_response: str (full response)
        """
        # Try LLM judge first if available
        if self.use_llm and self.judge:
            try:
                # Use batch_judge_answers with single comparison
                comparisons = [(model_answer, ground_truth)]
                results = self.judge.batch_judge_answers(comparisons)
                is_correct = results[0] if results else False

                return {
                    'is_correct': is_correct,
                    'confidence': 'high' if is_correct else 'medium',
                    'explanation': f'LLM judge evaluation: {is_correct}',
                    'judge_response': f'LLM judge result: {is_correct}',
                    'success': True,
                    'error_message': '',
                    'method_used': 'llm_judge'
                }

            except Exception as e:
                logger.debug(f"⚠️ LLM judge failed: {e}, falling back to string matching")
                # Fall through to string matching

        # Use string-based fallback
        result = StringBasedMatcher.string_match_answers(ground_truth, model_answer)
        result['method_used'] = 'string_fallback'
        return result

    def evaluate_baseline_and_variant(self, ground_truth: str,
                                    baseline_answer: str, variant_answer: str,
                                    baseline_problem: str, variant_problem: str) -> Dict[str, Any]:
        """
        Evaluate both baseline and variant answers, compute drift metrics.

        Returns:
            Dict with evaluation results and drift metrics
        """
        # Check baseline answer
        baseline_result = self.check_answer_match(
            ground_truth=ground_truth,
            model_answer=baseline_answer,
            problem_context=baseline_problem
        )

        # Check variant answer
        variant_result = self.check_answer_match(
            ground_truth=ground_truth,
            model_answer=variant_answer,
            problem_context=variant_problem
        )

        # Determine drift
        baseline_correct = baseline_result['is_correct']
        variant_correct = variant_result['is_correct']

        positive_drift = variant_correct and not baseline_correct  # Variant correct, baseline wrong
        negative_drift = not variant_correct and baseline_correct  # Variant wrong, baseline correct

        return {
            # Baseline evaluation
            'baseline_matches_ground_truth': baseline_correct,
            'baseline_confidence': baseline_result['confidence'],
            'baseline_explanation': baseline_result['explanation'],
            'baseline_judge_response': baseline_result['judge_response'],
            'baseline_evaluation_success': baseline_result['success'],
            'baseline_evaluation_error': baseline_result['error_message'],
            'baseline_evaluation_method': baseline_result.get('method_used', 'unknown'),

            # Variant evaluation
            'variant_matches_ground_truth': variant_correct,
            'variant_confidence': variant_result['confidence'],
            'variant_explanation': variant_result['explanation'],
            'variant_judge_response': variant_result['judge_response'],
            'variant_evaluation_success': variant_result['success'],
            'variant_evaluation_error': variant_result['error_message'],
            'variant_evaluation_method': variant_result.get('method_used', 'unknown'),

            # Drift metrics
            'positive_drift': positive_drift,  # Variant correct, baseline wrong
            'negative_drift': negative_drift,  # Variant wrong, baseline correct
            'has_drift': positive_drift or negative_drift,
            'baseline_variant_consistent': baseline_correct == variant_correct,

            # Overall evaluation success
            'evaluation_completed': baseline_result['success'] and variant_result['success']
        }


def test_answer_matcher():
    """Test the answer matcher functionality."""
    matcher = LLMAnswerMatcher()

    # Test case
    ground_truth = "42"
    baseline_answer = "The answer is 42"
    variant_answer = "I think it's 43"
    baseline_problem = "What is 6 * 7?"
    variant_problem = "What is six multiplied by seven?"

    result = matcher.evaluate_baseline_and_variant(
        ground_truth=ground_truth,
        baseline_answer=baseline_answer,
        variant_answer=variant_answer,
        baseline_problem=baseline_problem,
        variant_problem=variant_problem
    )

    logger.debug("Test Results:")
    logger.debug(f"Baseline correct: {result['baseline_matches_ground_truth']}")
    logger.debug(f"Variant correct: {result['variant_matches_ground_truth']}")
    logger.debug(f"Positive drift: {result['positive_drift']}")
    logger.debug(f"Negative drift: {result['negative_drift']}")
    logger.debug(f"Has drift: {result['has_drift']}")


if __name__ == "__main__":
    test_answer_matcher()