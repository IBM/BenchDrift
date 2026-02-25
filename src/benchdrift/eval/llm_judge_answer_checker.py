#!/usr/bin/env python3
"""
LLM-as-Judge Answer Checker

Uses an open-source LLM via RITS client to judge whether predicted answers 
match ground truth answers, even when representations differ.
"""

import json
import logging
import time
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import os

# Add the project root to path for imports

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations


logger = logging.getLogger('BenchDrift')
class LLMJudgeAnswerChecker:
    """
    Uses an LLM to judge answer equivalence with sophisticated prompting.
    """
    
    def __init__(self,
                 client_type: str = 'rits',
                 model_name: str = 'llama_3_3_70b',
                 batch_size: int = 50,
                 max_new_tokens: int = 10,
                 temperature: float = 0.0):
        """Initialize the LLM judge checker"""
        self.client_type = client_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Initialize model client
        self.model_client = self._initialize_model_client()
        
        logger.debug(f"✅ Initialized LLMJudgeAnswerChecker")
        logger.debug(f"   Client: {client_type}")
        logger.debug(f"   Model: {model_name}")
        logger.debug(f"   Batch size: {batch_size}")
    
    def _initialize_model_client(self):
        """Initialize model client"""
        try:
            return create_model_client_for_variations(
                client_type=self.client_type,
                model_name=self.model_name
            )
        except Exception as e:
            logger.debug(f"❌ Failed to initialize model client: {e}")
            raise
    
    def create_judge_prompt(self, predicted_answer: str, ground_truth: str,
                            few_shot_examples: str = "") -> str:
        """Create prompt for LLM judge to compare answers.

        Args:
            predicted_answer: The model's answer.
            ground_truth: The expected correct answer.
            few_shot_examples: Category-specific few-shot examples block.
                If empty, uses a minimal default.
        """
        if not few_shot_examples:
            few_shot_examples = (
                "- \"42\" vs \"42.0\" → YES (same value)\n"
                "- \"Monday\" vs \"Tuesday\" → NO (different days)\n"
                "- \"Option A\" vs \"Option B\" → NO (different options)\n"
                "- \"USA\" vs \"United States\" → YES (same country)"
            )

        prompt = f"""You are an expert answer evaluator. Determine if the predicted answer matches the ground truth.

GROUND TRUTH: {ground_truth}

PREDICTED ANSWER: {predicted_answer}

Rules:
1. Extract the core answer from the predicted text — ignore explanations, reasoning, and extra detail.
2. Compare the core answer to the ground truth for SEMANTIC EQUIVALENCE.
3. Ignore formatting differences: spacing, punctuation, capitalization, notation style.
4. If the answer contains JSON like {{"answer": "X"}}, extract the "answer" field value only.

{few_shot_examples}

Answer with ONLY "YES" or "NO" - nothing else."""

        return prompt
    
    def batch_judge_answers(self, comparisons: List[Tuple[str, str]],
                            dataset_name: str = "") -> List[bool]:
        """Judge a batch of answer comparisons.

        Args:
            comparisons: List of (predicted_answer, ground_truth) tuples.
            dataset_name: Optional HF dataset name for category-aware few-shots.
        """
        from benchdrift.eval.judge_few_shots import get_few_shot_examples

        # Prepare separate system and user prompts for batching
        system_prompts = []
        user_prompts = []

        for predicted, ground_truth in comparisons:
            few_shots = get_few_shot_examples(ground_truth, dataset_name)

            system_prompt = f"""You are an expert answer evaluator. Determine if the predicted answer matches the ground truth.

Rules:
1. Extract the core answer from the predicted text — ignore explanations, reasoning, and extra detail.
2. Compare the core answer to the ground truth for SEMANTIC EQUIVALENCE.
3. Ignore formatting differences: spacing, punctuation, capitalization, notation style.
4. If the answer contains JSON like {{"answer": "X"}}, extract the "answer" field value only.

{few_shots}

Answer with ONLY "YES" or "NO" - nothing else."""

            user_prompt = f"""GROUND TRUTH: {ground_truth}

PREDICTED ANSWER: {predicted}

Answer with ONLY "YES" or "NO" - nothing else."""

            system_prompts.append(system_prompt)
            user_prompts.append(user_prompt)

        # BATCH CALL: Process all comparisons at once
        try:
            logger.debug(f"        📦 Batching {len(comparisons)} judge prompts...")

            # Use batched model response for all RITS clients
            responses = self.model_client.get_model_response(
                system_prompts,
                user_prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            
            # Parse responses to boolean
            judgments = []
            for response in responses:
                # Extract YES/NO — use first word to avoid substring matches
                # (e.g., "YESTERDAY" contains "YES" but isn't an affirmative)
                response_clean = response.strip().upper()
                first_word = response_clean.split()[0] if response_clean.split() else ""

                if first_word == 'YES' or response_clean == 'YES':
                    judgments.append(True)
                elif first_word == 'NO' or response_clean == 'NO':
                    judgments.append(False)
                elif response_clean.startswith('YES'):
                    judgments.append(True)
                elif response_clean.startswith('NO'):
                    judgments.append(False)
                else:
                    # Fallback: if unclear, mark as incorrect (conservative)
                    logger.debug(f"Unclear judge response: {response[:50]}... -> Marking as NO")
                    judgments.append(False)
            
            return judgments
            
        except Exception as e:
            logger.debug(f"❌ Error in batch judging: {e}")
            # Return all False if there's an error (conservative approach)
            return [False] * len(comparisons)
    
    def judge_all_answers(self, responses_file: str, ground_truth_file: str) -> Dict:
        """Judge all answers using LLM and save results"""
        
        logger.debug(f"📖 Loading ground truth from {ground_truth_file}...")
        
        # Load ground truth
        ground_truth = {}
        with open(ground_truth_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        problem_id = data.get('id')
                        answer = data.get('answer', '')
                        if problem_id and answer:
                            ground_truth[problem_id] = answer
                    except:
                        continue
        
        logger.debug(f"✅ Loaded ground truth for {len(ground_truth)} problems")
        
        # Collect all comparisons
        logger.debug(f"📖 Processing responses from {responses_file}...")
        
        all_comparisons = []
        response_metadata = []
        
        with open(responses_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Get basic info
                    variation_id = data.get('variation_id', '')
                    final_answer = data.get('final_answer', '')
                    is_baseline = data.get('is_baseline', False)
                    success = data.get('success', False)
                    
                    # Skip failed responses
                    if not success or not final_answer.strip():
                        continue
                    
                    # Extract problem ID
                    if is_baseline:
                        problem_id = variation_id
                    else:
                        # For variations, extract problem ID from variation_id
                        if '_' in variation_id:
                            problem_id = '_'.join(variation_id.split('_')[:-1])
                        else:
                            problem_id = variation_id
                    
                    # Get ground truth for this problem
                    gt_answer = ground_truth.get(problem_id, '')
                    
                    if not gt_answer:
                        continue
                    
                    # Add to comparison batch
                    all_comparisons.append((final_answer, gt_answer))
                    response_metadata.append({
                        'variation_id': variation_id,
                        'problem_id': problem_id,
                        'is_baseline': is_baseline,
                        'final_answer': final_answer,
                        'ground_truth': gt_answer,
                        'line_num': line_num
                    })
                    
                except Exception as e:
                    logger.debug(f"⚠️  Error processing line {line_num}: {e}")
                    continue
        
        logger.debug(f"✅ Collected {len(all_comparisons)} comparisons for LLM judging")
        
        # Process in batches
        all_judgments = []
        total_batches = (len(all_comparisons) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(all_comparisons), self.batch_size):
            batch_num = i // self.batch_size + 1
            logger.debug(f"🔄 Processing batch {batch_num}/{total_batches} ({self.batch_size} comparisons)...")
            
            batch_comparisons = all_comparisons[i:i + self.batch_size]
            batch_judgments = self.batch_judge_answers(batch_comparisons)
            all_judgments.extend(batch_judgments)
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
        
        # Combine results with metadata
        baseline_results = {}
        variation_results = {}
        
        for metadata, judgment in zip(response_metadata, all_judgments):
            metadata['correct'] = judgment
            
            if metadata['is_baseline']:
                baseline_results[metadata['problem_id']] = metadata
            else:
                problem_id = metadata['problem_id']
                if problem_id not in variation_results:
                    variation_results[problem_id] = []
                variation_results[problem_id].append(metadata)
        
        # Calculate statistics
        baseline_correct = sum(1 for result in baseline_results.values() if result['correct'])
        baseline_total = len(baseline_results)
        baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0
        
        variation_correct = sum(1 for variations in variation_results.values() 
                              for var in variations if var['correct'])
        variation_total = sum(len(variations) for variations in variation_results.values())
        variation_accuracy = variation_correct / variation_total if variation_total > 0 else 0
        
        drift = baseline_accuracy - variation_accuracy
        relative_drift = (drift / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        
        logger.debug(f"\n📊 LLM JUDGE RESULTS:")
        logger.debug(f"=====================")
        logger.debug(f"BASELINE RESULTS:")
        logger.debug(f"  Total: {baseline_total}")
        logger.debug(f"  Correct: {baseline_correct}")
        logger.debug(f"  Accuracy: {baseline_accuracy:.3f}")
        
        logger.debug(f"\nVARIATION RESULTS:")
        logger.debug(f"  Total: {variation_total}")
        logger.debug(f"  Correct: {variation_correct}")
        logger.debug(f"  Accuracy: {variation_accuracy:.3f}")
        
        logger.debug(f"\nDRIFT ANALYSIS:")
        logger.debug(f"  Accuracy Drift: {drift:.3f}")
        logger.debug(f"  Relative Drift: {relative_drift:.1f}%")
        
        # Save detailed results
        results = {
            'summary': {
                'baseline_accuracy': baseline_accuracy,
                'variation_accuracy': variation_accuracy,
                'drift': drift,
                'relative_drift': relative_drift,
                'baseline_total': baseline_total,
                'variation_total': variation_total,
                'baseline_correct': baseline_correct,
                'variation_correct': variation_correct,
                'model_used': self.model_name,
                'batch_size': self.batch_size
            },
            'baseline_results': baseline_results,
            'variation_results': variation_results
        }
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save LLM judge results to file"""
        logger.debug(f"\n💾 Saving LLM judge results to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug(f"✅ Results saved!")

def main():
    responses_file = "temporal_subset_100_with_variations_responses.jsonl"
    ground_truth_file = "temporal_subset_100_with_variations.jsonl"
    output_file = "llm_judge_results.json"
    
    if not Path(responses_file).exists():
        logger.debug(f"❌ Responses file not found: {responses_file}")
        return 1
    
    if not Path(ground_truth_file).exists():
        logger.debug(f"❌ Ground truth file not found: {ground_truth_file}")
        return 1
    
    logger.debug("🤖 LLM-AS-JUDGE ANSWER CHECKER")
    logger.debug("===============================")
    
    # Initialize checker
    try:
        checker = LLMJudgeAnswerChecker(
            client_type='rits',
            model_name='llama_3_3_70b',
            batch_size=50,  # Process 50 comparisons at a time
            max_new_tokens=10,  # Just need YES/NO
            temperature=0.0   # Maximum determinism for consistent judgments
        )
    except Exception as e:
        logger.debug(f"❌ Failed to initialize LLM judge: {e}")
        return 1
    
    # Judge all answers
    try:
        results = checker.judge_all_answers(responses_file, ground_truth_file)
        checker.save_results(results, output_file)
        
        logger.debug(f"\n🎉 LLM judge analysis complete!")
        logger.debug(f"   Baseline Accuracy: {results['summary']['baseline_accuracy']:.1%}")
        logger.debug(f"   Variation Accuracy: {results['summary']['variation_accuracy']:.1%}")
        logger.debug(f"   Drift: {results['summary']['relative_drift']:.1f}%")
        logger.debug(f"   Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.debug(f"❌ Error during LLM judging: {e}")
        return 1

if __name__ == "__main__":
    exit(main())