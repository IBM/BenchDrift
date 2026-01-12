#!/usr/bin/env python3
"""
Complete Unified Variation Pipeline - Progressive Enhancement

Unified approach where each stage enhances the same JSON file:
1. Stage 1: Generate variations -> adds variation data
2. Stage 2: Generate responses -> adds model response data
3. Stage 3: Evaluation -> adds evaluation metrics

Each row contains: baseline question, variant question, transformation details,
model responses, and evaluation metrics - all in one place.

No more multiple intermediate files - everything progressively enhanced in one unified file.
"""

import os
from tqdm import tqdm
import logging
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
import pandas as pd
from datetime import datetime
import math

# Add current directory to path for imports

sys.path.append(str(Path(__file__).parent))

from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
from benchdrift.pipeline.variation_answer_generator import VariationAnswerGenerator, VariationProblem
from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations
from benchdrift.eval.llm_answer_matcher import LLMAnswerMatcher
# No longer need separate enhancement script - all done progressively



logger = logging.getLogger('BenchDrift')
class UnifiedProgressivePipeline:
    """Unified progressive enhancement pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.unified_file = config['unified_file']
        self.data = []
        self.batch_size = config.get('batch_size', 50)

        # Load existing data if file exists
        if Path(self.unified_file).exists():
            with open(self.unified_file, 'r') as f:
                self.data = json.load(f)
            logger.debug(f"📂 Loaded {len(self.data)} existing entries from {self.unified_file}")
    
    def save_data(self, stage_name: str):
        """Save current data state with stage tracking."""
        # Add metadata about processing stages
        all_stages = set()
        for entry in self.data:
            if 'stages_completed' in entry:
                all_stages.update(entry.get('stages_completed', []))

        metadata = {
            'last_updated': datetime.now().isoformat(),
            'last_stage': stage_name,
            'total_entries': len(self.data),
            'stages_completed': sorted(list(all_stages))
        }

        # Save main data
        try:
            with open(self.unified_file, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"❌ Error saving unified file: {e}")
            # Try to save with more robust serialization
            try:
                with open(self.unified_file, 'w') as f:
                    json.dump(self.data, f, indent=2, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x)
            except Exception as e2:
                logger.debug(f"❌ Critical error saving data: {e2}")
                raise

        # Save metadata separately
        metadata_file = self.unified_file.replace('.json', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"💾 Saved {len(self.data)} entries after {stage_name} to {self.unified_file}")

    def stage1_generate_variations(self):
        """Stage 1: Generate variations and create unified entries."""
        logger.debug(f"\n🔄 Stage 1: Generating Variations...")

        # Load input problems OR use test example
        input_file = self.config.get('input_problems')

        if not input_file or input_file == 'test':
            # Use hardcoded test example
            problems = [{"problem": "What is 15 + 25?", "answer": "40"}]
            logger.debug(f"🧪 Using test example: {problems[0]['problem']}")
        else:
            # Load from file
            with open(input_file, 'r') as f:
                if input_file.endswith('.json'):
                    data = json.load(f)
                    # Handle both single problem dict and list of problems
                    if isinstance(data, dict):
                        # Single problem object - wrap in list
                        problems = [data]
                    elif isinstance(data, list):
                        # List of problems
                        problems = data
                    else:
                        # String or other type - treat as single problem
                        problems = [{"problem": str(data)}]
                else:  # jsonl
                    problems = [json.loads(line) for line in f if line.strip()]

        logger.debug(f"📂 Loaded {len(problems)} input problems")

        # Initialize variation engine
        if self.config.get('use_model_client', True):
            model_client = create_model_client_for_variations(
                client_type=self.config.get('client_type', 'rits'),
                model_name=self.config.get('model_name', 'mistral_small_3_2_instruct')
            )
        else:
            model_client = None

        engine = UnifiedVariationEngine(model_client=model_client)

        # Process in batches if specified
        if self.config.get('batch_size', 0) > 0:
            self._process_variations_batched(problems, engine)
        else:
            self._process_variations_sequential(problems, engine)

        self.save_data("variations")
        logger.debug(f"✅ Stage 1 complete: Generated {len(self.data)} total entries")

    def _process_variations_sequential(self, problems: List, engine):
        """Process variations sequentially."""
        for i, problem in enumerate(problems):
            logger.debug(f"🔄 Processing problem {i+1}/{len(problems)}")
            self._create_problem_entries(i, problem, engine)

    def _process_variations_batched(self, problems: List, engine):
        """Process variations in batches."""
        num_batches = math.ceil(len(problems) / self.batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(problems))
            batch_problems = problems[start_idx:end_idx]

            logger.debug(f"\n📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_problems)} problems)")

            for i, problem in enumerate(batch_problems):
                actual_idx = start_idx + i
                logger.debug(f"  🔄 Processing problem {actual_idx + 1}")
                self._create_problem_entries(actual_idx, problem, engine)

            # Save after each batch
            if self.config.get('save_every_batch', True):
                self.save_data(f"variations_batch_{batch_idx + 1}")
                logger.debug(f"💾 Saved after batch {batch_idx + 1}")

    def _create_problem_entries_direct(self, problem: dict):
        """Create entries directly from a problem dict - for testing without file input."""
        from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine

        # Initialize engine
        if self.config.get('use_model_client', True):
            model_client = create_model_client_for_variations(
                client_type=self.config.get('client_type', 'rits'),
                model_name=self.config.get('model_name', 'mistral_small_3_2_instruct')
            )
        else:
            model_client = None

        engine = UnifiedVariationEngine(model_client=model_client)

        # Process the single problem
        self._create_problem_entries(0, problem, engine)

    def _create_problem_entries(self, i: int, problem: Any, engine):
        """Create baseline and variant entries for a problem."""
        # Extract problem data
        if isinstance(problem, dict):
            problem_text = problem.get('problem', problem.get('question', str(problem)))
            problem_id = problem.get('id', f'problem_{i}')
            ground_truth = problem.get('answer', problem.get('ground_truth', ''))
        else:
            problem_text = str(problem)
            problem_id = f'problem_{i}'
            ground_truth = ''

        # Generate variations
        try:
            # Generate variations - UnifiedVariationEngine returns a dict organized by category
            variations_dict = engine.generate_comprehensive_variations(
                problem_text,
                organize_by="debugging_capabilities"
            )

            # Flatten the organized variations dict to a list
            variations = []
            for category, vars_list in variations_dict.items():
                for var in vars_list:
                    variations.append(var)

            timestamp = datetime.now().isoformat()

            # Create baseline entry
            baseline_entry = self._create_baseline_entry(problem_id, problem_text, ground_truth, timestamp)
            self.data.append(baseline_entry)

            # Create variant entries
            for j, variation in enumerate(variations):
                variant_entry = self._create_variant_entry(problem_id, problem_text, ground_truth, variation, j, len(variations), timestamp)
                self.data.append(variant_entry)

        except Exception as e:
            logger.debug(f"    ❌ Error processing problem {i+1}: {e}")
            return

    def _create_baseline_entry(self, problem_id: str, problem_text: str, ground_truth: str, timestamp: str) -> Dict[str, Any]:
        """Create baseline entry."""
        return {
            'problem_id': problem_id,
            'variation_id': f'{problem_id}_baseline',
            'variation_index': 0,
            'variation_type': 'baseline',
            'is_baseline': True,
            'is_variant': False,

            # Problem content
            'original_problem': problem_text,
            'modified_problem': problem_text,
            'baseline_problem': problem_text,
            'ground_truth_answer': ground_truth,

            # Transformation details (baseline values)
            'transformation_type': 'baseline',
            'original_component': 'baseline',
            'new_component': 'baseline',
            'debugging_capability': 'baseline',
            'generation_method': 'baseline',
            'detection_method': 'baseline',
            'domains_involved': ['baseline'],
            'combination_size': 0,
            'cross_domain': False,
            'confidence': 'baseline',

            # Stage tracking
            'stages_completed': ['variations'],
            'variation_generation_timestamp': timestamp,

            # Response placeholders (to be filled in Stage 2)
            'has_model_response': False,
            'model_response': '',
            'model_thinking': '',
            'model_final_answer': '',
            'response_generation_time': 0.0,
            'response_success': False,
            'response_error_message': '',
            'response_timestamp': 0,

            # Evaluation placeholders (to be filled in Stage 3)
            'baseline_model_answer': '',
            'baseline_model_thinking': '',
            'baseline_model_response': '',
            'baseline_response_success': False,
            'baseline_matches_ground_truth': False,
            'variant_matches_ground_truth': False,
            'baseline_variant_consistent': False,
            'has_drift': False,
            'has_improvement': False
        }

    def _create_variant_entry(self, problem_id: str, problem_text: str, ground_truth: str, variation: Dict, j: int, total_variations: int, timestamp: str) -> Dict[str, Any]:
        """Create variant entry with all necessary fields."""
        return {
            # Problem identification
            'problem_id': problem_id,
            'variation_id': f'{problem_id}_variant_{j+1}',
            'variation_index': j + 1,
            'variation_type': 'variant',
            'is_baseline': False,
            'is_variant': True,
            'variant_number': j + 1,
            'total_variants_for_problem': total_variations,
            'problem_index': int(problem_id.split('_')[-1]) if '_' in problem_id else j,

            # Problem content
            'original_problem': problem_text,
            'modified_problem': variation.get('modified_problem', ''),
            'baseline_problem': problem_text,
            'ground_truth_answer': ground_truth,

            # Transformation details from variation (complete set)
            'transformation_type': variation.get('transformation_type', ''),
            'original_component': variation.get('original_component', ''),
            'new_component': variation.get('new_component', ''),
            'debugging_capability': variation.get('debugging_capability', ''),
            'generation_method': variation.get('generation_method', ''),
            'detection_method': variation.get('detection_method', ''),
            'domains_involved': variation.get('domains_involved', []),
            'combination_size': variation.get('combination_size', 0),
            'cross_domain': variation.get('cross_domain', False),
            'confidence': variation.get('confidence', ''),
            'candidates_transformed': variation.get('candidates_transformed', []),
            'selected_variants': variation.get('selected_variants', []),
            'transformation_details': variation.get('transformation_details', {}),

            # Additional metadata fields
            'variation_complexity': variation.get('variation_complexity', 'medium'),
            'transformation_confidence': variation.get('transformation_confidence', 1.0),
            'semantic_similarity': variation.get('semantic_similarity', 0.0),
            'lexical_overlap': variation.get('lexical_overlap', 0.0),

            # Stage tracking
            'stages_completed': ['variations'],
            'variation_generation_timestamp': timestamp,

            # Response placeholders - baseline answers (to be filled in Stage 2)
            'baseline_answer': '',  # Model answer to baseline question
            'baseline_thinking': '',
            'baseline_response': '',
            'baseline_generation_time': 0.0,
            'baseline_success': False,
            'baseline_error_message': '',

            # Response placeholders - variant answers (to be filled in Stage 2)
            'variant_answer': '',  # Model answer to variant question
            'variant_thinking': '',
            'variant_response': '',
            'variant_generation_time': 0.0,
            'variant_success': False,
            'variant_error_message': '',

            # General response tracking
            'has_model_response': False,
            'response_timestamp': 0,

            # Evaluation placeholders (to be filled in Stage 3)
            'baseline_matches_ground_truth': False,
            'variant_matches_ground_truth': False,
            'baseline_variant_consistent': False,
            'has_drift': False,
            'has_improvement': False,
            'answer_similarity_score': 0.0,
            'evaluation_method': '',
            'evaluation_timestamp': '',

            # LLM Judge evaluation placeholders
            'llm_judge_baseline_correct': False,
            'llm_judge_variant_correct': False,
            'llm_judge_consistent': False,
            'llm_judge_confidence': 0.0,
            'llm_judge_reasoning': ''
        }
    
    def stage2_generate_responses(self, eval_model=None, force_regenerate=False):
        """Stage 2: Generate model responses - optimized to avoid duplicate baseline calls."""
        logger.debug(f"\n🔄 Stage 2: Generating Model Responses (Optimized)...")

        if eval_model is not None:
                model_name = eval_model
        else:
                model_name=self.config.get('response_model', 'mistral_small_3_2_instruct')

        # Use pre-initialized BaseModelClient if provided, otherwise create VariationAnswerGenerator
        from benchdrift.models.model_client import BaseModelClient

        if isinstance(model_name, BaseModelClient):
            logger.debug(f"✅ Using pre-initialized model client: {type(model_name).__name__}")
            answer_gen_like = model_name
            answer_gen = self._create_adapter_wrapper(answer_gen_like)
        else:
            client_type = self.config.get('client_type', 'rits')
            response_batch_size = self.config.get('response_eval_batch_size', self.batch_size)
            answer_gen = VariationAnswerGenerator(
                client_type=client_type,
                model_name=model_name,
                max_workers=self.config.get('max_workers', 4),
                batch_size=response_batch_size,  # Use response-specific batch size
                rits_batch_size=self.config.get('rits_batch_size', min(response_batch_size, 10)),  # RITS batch size
                max_new_tokens=self.config.get('max_new_tokens', 1024),  # Maximum tokens to generate
                temperature=self.config.get('temperature', 0.1),  # Temperature for generation
                disable_cot=self.config.get('disable_cot', False)  # Disable CoT for faster generation
            )

        # Find entries that need responses
        if force_regenerate or self.config.get('force_regenerate', False):
            # Force regeneration: process all entries with variations, even if responses exist
            entries_needing_responses = [
                entry for entry in self.data
                if 'variations' in entry.get('stages_completed', [])
            ]
            logger.debug(f"   🔄 FORCE REGENERATE: Processing all {len(entries_needing_responses)} entries (ignoring existing responses)")
        else:
            # Normal operation: skip entries that already have responses
            entries_needing_responses = [
                entry for entry in self.data
                if 'variations' in entry.get('stages_completed', [])
                and 'responses' not in entry.get('stages_completed', [])
            ]

            # Check if there are entries with responses already completed
            entries_with_responses = [
                entry for entry in self.data
                if 'responses' in entry.get('stages_completed', [])
            ]

            if entries_with_responses:
                logger.debug(f"   ℹ️  Found {len(entries_with_responses)} entries with existing responses (skipping)")
                logger.debug(f"   💡 To regenerate responses, use --force-regenerate or set force_regenerate=True")


        logger.debug(f"📝 Generating responses for {len(entries_needing_responses)} entries...")
        logger.debug(f"   🚀 OPTIMIZATION: Generate baseline answers once per problem, reuse across variants")

        # Group entries by problem_id to optimize baseline generation
        problem_groups = {}
        for entry in entries_needing_responses:
            problem_id = entry['problem_id']
            if problem_id not in problem_groups:
                problem_groups[problem_id] = {'baseline': None, 'variants': []}

            if entry.get('is_baseline', False):
                problem_groups[problem_id]['baseline'] = entry
            else:
                problem_groups[problem_id]['variants'].append(entry)

        logger.debug(f"   📊 Found {len(problem_groups)} unique problems")

        # Step 1: Collect all problems that need responses (exploiting batch processing)
        all_problems_to_process = []
        problem_id_mapping = {}  # Maps variation_id to (problem_id, entry_type, entry)

        for problem_id, group in problem_groups.items():
            # Add baseline problem
            baseline_entry = group['baseline']
            if baseline_entry:
                baseline_variation_id = f"{problem_id}_baseline"
                baseline_problem = VariationProblem(
                    variation_id=baseline_variation_id,
                    original_problem=baseline_entry['original_problem'],
                    modified_problem=baseline_entry['original_problem'],
                    transformation_type='baseline',
                    original_component='',
                    new_component='',
                    metadata={'entry': baseline_entry}
                )
                all_problems_to_process.append(baseline_problem)
                problem_id_mapping[baseline_variation_id] = (problem_id, 'baseline', baseline_entry)

            # Add variant problems
            for variant_entry in group['variants']:
                variant_variation_id = variant_entry['variation_id']
                variant_problem = VariationProblem(
                    variation_id=variant_variation_id,
                    original_problem=variant_entry['original_problem'],
                    modified_problem=variant_entry['modified_problem'],
                    transformation_type=variant_entry.get('transformation_type', ''),
                    original_component=variant_entry.get('original_component', ''),
                    new_component=variant_entry.get('new_component', ''),
                    metadata={'entry': variant_entry}
                )
                all_problems_to_process.append(variant_problem)
                problem_id_mapping[variant_variation_id] = (problem_id, 'variant', variant_entry)

        logger.debug(f"   🚀 BATCH PROCESSING: Generating responses for {len(all_problems_to_process)} problems in batches...")

        # Step 2: Generate all responses in batches (this is where the magic happens!)
        try:
            all_responses = answer_gen.generate_responses_parallel(all_problems_to_process)
            logger.debug(f"   ✅ Generated {len(all_responses)} responses using batch processing")

            # Step 3: Process responses and organize into baseline cache + variant updates
            baseline_answers_cache = {}
            response_dict = {resp.variation_id: resp for resp in all_responses}

            for variation_id, response in response_dict.items():
                problem_id, entry_type, entry = problem_id_mapping[variation_id]

                # Convert response to data format
                response_data = {
                    'model_response': response.model_response,
                    'model_thinking': response.model_response,  # Store full raw response (parsing may fail)
                    'model_final_answer': response.final_answer,
                    'response_generation_time': response.generation_time,
                    'response_success': response.success,
                    'response_error_message': response.error_message or '',
                    'response_timestamp': response.timestamp
                }

                if entry_type == 'baseline':
                    # Cache baseline answer for reuse across variants
                    baseline_answers_cache[problem_id] = response_data
                    self._update_entry_with_response_data(entry, response_data, is_baseline=True)
                    logger.debug(f"     ✅ Baseline cached for {problem_id}")
                else:
                    # Update variant entry
                    self._update_entry_with_response_data(entry, response_data, is_baseline=False)
                    # Add cached baseline data
                    if problem_id in baseline_answers_cache:
                        self._add_baseline_data_to_variant(entry, baseline_answers_cache[problem_id])
                    logger.debug(f"     ✅ Variant processed for {variation_id}")

        except Exception as e:
            logger.debug(f"❌ Batch processing failed: {e}")
            # Fallback to individual processing if batch fails
            logger.debug("   🔄 Falling back to individual processing...")
            baseline_answers_cache = {}

            for problem_id, group in problem_groups.items():
                logger.debug(f"\n   🔄 Processing problem: {problem_id}")

                # Generate baseline answer once per problem
                baseline_entry = group['baseline']
                if baseline_entry:
                    baseline_answer_data = self._generate_single_response(
                        baseline_entry['original_problem'],
                        answer_gen,
                        f"{problem_id}_baseline"
                    )
                    baseline_answers_cache[problem_id] = baseline_answer_data
                    self._update_entry_with_response_data(baseline_entry, baseline_answer_data, is_baseline=True)
                    logger.debug(f"     ✅ Baseline answer generated and cached")

                # Generate variant answers and reuse baseline
                variants = group['variants']
                logger.debug(f"     🔄 Generating {len(variants)} variant answers...")

                for variant_entry in variants:
                    variant_answer_data = self._generate_single_response(
                        variant_entry['modified_problem'],
                        answer_gen,
                        variant_entry['variation_id']
                    )
                    self._update_entry_with_response_data(variant_entry, variant_answer_data, is_baseline=False)
                    if problem_id in baseline_answers_cache:
                        self._add_baseline_data_to_variant(variant_entry, baseline_answers_cache[problem_id])

        self.save_data("responses")
        logger.debug(f"✅ Stage 2 complete: Added optimized responses to unified file")

    def _create_adapter_wrapper(self, model_client):
        """Wrap BaseModelClient to provide VariationAnswerGenerator interface."""
        import time
        from dataclasses import dataclass

        @dataclass
        class GeneratedResponse:
            variation_id: str
            problem: str
            model_response: str
            thinking: str
            final_answer: str
            generation_time: float
            success: bool
            error_message: Optional[str] = None
            timestamp: float = 0.0

            def __post_init__(self):
                if self.timestamp == 0.0:
                    self.timestamp = time.time()

        class AdapterWrapper:
            def __init__(self, client):
                self.model_client = client

            def generate_responses_parallel(self, variations):
                responses = []
                for var in variations:
                    try:
                        # Create minimal system/user prompt split
                        problem_text = var.modified_problem
                        system_prompt = ""
                        user_prompt = problem_text

                        # Get response from model client
                        response_text = self.model_client.get_single_response(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            max_new_tokens=1024,
                            temperature=0.1
                        )

                        responses.append(GeneratedResponse(
                            variation_id=var.variation_id,
                            problem=problem_text,
                            model_response=response_text,
                            thinking="",
                            final_answer=response_text,
                            generation_time=0.0,
                            success=True,
                            error_message=None
                        ))
                    except Exception as e:
                        logger.debug(f"❌ Failed to generate response for {var.variation_id}: {e}")
                        responses.append(GeneratedResponse(
                            variation_id=var.variation_id,
                            problem=var.modified_problem,
                            model_response="",
                            thinking="",
                            final_answer="",
                            generation_time=0.0,
                            success=False,
                            error_message=str(e)
                        ))
                return responses

        return AdapterWrapper(model_client)

    def _generate_single_response(self, problem_text: str, answer_gen: VariationAnswerGenerator, variation_id: str) -> Dict[str, Any]:
        try:
            problem = VariationProblem(
                variation_id=variation_id,
                original_problem=problem_text,
                modified_problem=problem_text,
                transformation_type='',
                original_component='',  # Required parameter
                new_component='',       # Required parameter
                metadata={}
            )

            responses = answer_gen.generate_responses_parallel([problem])


            if responses:
                response = responses[0]
                return {
                    'model_response': response.model_response,
                    'model_thinking': response.model_response,  # Store full raw response (parsing may fail)
                    'model_final_answer': response.final_answer,
                    'response_generation_time': response.generation_time,
                    'response_success': response.success,
                    'response_error_message': response.error_message or '',
                    'response_timestamp': response.timestamp
                }
            else:
                return self._empty_response_data()
        except Exception as e:
            logger.debug(f"❌ Error generating response for {variation_id}: {e}")
            return self._empty_response_data()

    def _empty_response_data(self) -> Dict[str, Any]:
        return {
            'model_response': '',
            'model_thinking': '',
            'model_final_answer': '',
            'response_generation_time': 0.0,
            'response_success': False,
            'response_error_message': 'Failed to generate',
            'response_timestamp': 0
        }

    def _update_entry_with_response_data(self, entry: Dict[str, Any], response_data: Dict[str, Any], is_baseline: bool):
        if is_baseline:
            # For baseline entries, update baseline response fields
            entry.update({
                'baseline_answer': response_data['model_final_answer'],
                'baseline_thinking': response_data['model_thinking'],
                'baseline_response': response_data['model_response'],
                'baseline_generation_time': response_data['response_generation_time'],
                'baseline_response_success': response_data['response_success'],
                'baseline_response_error': response_data['response_error_message'],
                'baseline_response_timestamp': response_data['response_timestamp'],

                # For baseline entries, variant data is same as baseline
                'variant_answer': response_data['model_final_answer'],
                'variant_thinking': response_data['model_thinking'],
                'variant_response': response_data['model_response'],
                'variant_generation_time': response_data['response_generation_time'],
                'variant_response_success': response_data['response_success'],
                'variant_response_error': response_data['response_error_message'],
                'variant_response_timestamp': response_data['response_timestamp']
            })
        else:
            # For variant entries, update variant response fields
            entry.update({
                'variant_answer': response_data['model_final_answer'],
                'variant_thinking': response_data['model_thinking'],
                'variant_response': response_data['model_response'],
                'variant_generation_time': response_data['response_generation_time'],
                'variant_response_success': response_data['response_success'],
                'variant_response_error': response_data['response_error_message'],
                'variant_response_timestamp': response_data['response_timestamp']
            })

        # Add legacy fields for backward compatibility
        entry.update({
            'model_final_answer': response_data['model_final_answer'],
            'model_thinking': response_data['model_thinking'],
            'model_response': response_data['model_response'],
            'response_generation_time': response_data['response_generation_time'],
            'response_success': response_data['response_success'],
            'response_error_message': response_data['response_error_message'],
            'response_timestamp': response_data['response_timestamp'],
            'has_model_response': response_data['response_success']
        })

        # Mark response stage as complete
        if 'stages_completed' not in entry:
            entry['stages_completed'] = []
        if 'responses' not in entry['stages_completed']:
            entry['stages_completed'].append('responses')

    def _add_baseline_data_to_variant(self, variant_entry: Dict[str, Any], baseline_data: Dict[str, Any]):
        variant_entry.update({
            'baseline_answer': baseline_data['model_final_answer'],
            'baseline_thinking': baseline_data['model_thinking'],
            'baseline_response': baseline_data['model_response'],
            'baseline_generation_time': baseline_data['response_generation_time'],
            'baseline_response_success': baseline_data['response_success'],
            'baseline_response_error': baseline_data['response_error_message'],
            'baseline_response_timestamp': baseline_data['response_timestamp']
        })

    def _process_responses_sequential(self, entries_needing_responses: List[Dict], answer_gen):
        """Process responses sequentially - generate both baseline and variant answers."""

        # For each entry, we need to generate TWO responses:
        # 1. Response to baseline question (original_problem)
        # 2. Response to variant question (modified_problem)

        all_problems = []

        for entry in entries_needing_responses:
            # Create problem for baseline question
            baseline_problem = VariationProblem(
                variation_id=f"{entry['variation_id']}_baseline",
                original_problem=entry['original_problem'],
                modified_problem=entry['original_problem'],  # Same as original for baseline
                transformation_type='baseline',
                metadata={**entry, 'question_type': 'baseline'}
            )
            all_problems.append(baseline_problem)

            # Create problem for variant question (only for variant entries)
            if not entry.get('is_baseline', False):
                variant_problem = VariationProblem(
                    variation_id=f"{entry['variation_id']}_variant",
                    original_problem=entry['original_problem'],
                    modified_problem=entry['modified_problem'],
                    transformation_type=entry.get('transformation_type', ''),
                    metadata={**entry, 'question_type': 'variant'}
                )
                all_problems.append(variant_problem)

        logger.debug(f"   Generating {len(all_problems)} total responses ({len(entries_needing_responses)} baseline + variant pairs)")

        # Generate responses
        try:
            responses = answer_gen.generate_responses_for_variations(all_problems)
            self._update_entries_with_dual_responses(responses)
        except Exception as e:
            logger.debug(f"❌ Error generating responses: {e}")

    def _process_responses_batched(self, entries_needing_responses: List[Dict], answer_gen):
        """Process responses in batches."""
        num_batches = math.ceil(len(entries_needing_responses) / self.batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(entries_needing_responses))
            batch_entries = entries_needing_responses[start_idx:end_idx]

            logger.debug(f"\n📦 Processing response batch {batch_idx + 1}/{num_batches} ({len(batch_entries)} entries)")

            # Convert to VariationProblem objects
            variation_problems = []
            for entry in batch_entries:
                problem = VariationProblem(
                    variation_id=entry['variation_id'],
                    original_problem=entry['original_problem'],
                    modified_problem=entry['modified_problem'],
                    transformation_type=entry.get('transformation_type', ''),
                    metadata=entry
                )
                variation_problems.append(problem)

            # Generate responses for this batch
            try:
                responses = answer_gen.generate_responses_for_variations(variation_problems)
                self._update_entries_with_responses(responses)

                # Save after each batch
                if self.config.get('save_every_batch', True):
                    self.save_data(f"responses_batch_{batch_idx + 1}")
                    logger.debug(f"💾 Saved after batch {batch_idx + 1}")

            except Exception as e:
                logger.debug(f"    ❌ Error processing response batch: {e}")

    def _update_entries_with_responses(self, responses: List[Dict]):
        """Update entries with response data."""
        response_dict = {resp['variation_id']: resp for resp in responses}

        for entry in self.data:
            if entry['variation_id'] in response_dict:
                resp_data = response_dict[entry['variation_id']]

                # Update response fields
                entry.update({
                    'has_model_response': True,
                    'model_response': resp_data.get('model_response', ''),
                    'model_thinking': resp_data.get('model_response', ''),  # Store full raw response (parsing may fail)
                    'model_final_answer': resp_data.get('final_answer', ''),
                    'response_generation_time': resp_data.get('generation_time', 0.0),
                    'response_success': resp_data.get('success', False),
                    'response_error_message': resp_data.get('error_message', ''),
                    'response_timestamp': resp_data.get('timestamp', 0),
                    'response_generation_timestamp': datetime.now().isoformat()
                })

                # Update stage tracking
                if 'stages_completed' not in entry:
                    entry['stages_completed'] = []
                if 'responses' not in entry['stages_completed']:
                    entry['stages_completed'].append('responses')

    def _update_entries_with_dual_responses(self, responses: List[Dict]):
        """Update entries with both baseline and variant response data."""
        response_dict = {resp['variation_id']: resp for resp in responses}

        for entry in self.data:
            # Look for baseline response
            baseline_resp_id = f"{entry['variation_id']}_baseline"
            variant_resp_id = f"{entry['variation_id']}_variant"

            # Update baseline response fields
            if baseline_resp_id in response_dict:
                baseline_resp = response_dict[baseline_resp_id]
                entry.update({
                    'baseline_answer': baseline_resp.get('final_answer', ''),
                    'baseline_thinking': baseline_resp.get('model_response', ''),  # Store full raw response (parsing may fail)
                    'baseline_response': baseline_resp.get('model_response', ''),
                    'baseline_generation_time': baseline_resp.get('generation_time', 0.0),
                    'baseline_success': baseline_resp.get('success', False),
                    'baseline_error_message': baseline_resp.get('error_message', ''),
                })

            # Update variant response fields (only for variant entries)
            if not entry.get('is_baseline', False) and variant_resp_id in response_dict:
                variant_resp = response_dict[variant_resp_id]
                entry.update({
                    'variant_answer': variant_resp.get('final_answer', ''),
                    'variant_thinking': variant_resp.get('model_response', ''),  # Store full raw response (parsing may fail)
                    'variant_response': variant_resp.get('model_response', ''),
                    'variant_generation_time': variant_resp.get('generation_time', 0.0),
                    'variant_success': variant_resp.get('success', False),
                    'variant_error_message': variant_resp.get('error_message', ''),
                })
            elif entry.get('is_baseline', False):
                # For baseline entries, variant answer is same as baseline
                entry.update({
                    'variant_answer': entry.get('baseline_answer', ''),
                    'variant_thinking': entry.get('baseline_thinking', ''),
                    'variant_response': entry.get('baseline_response', ''),
                    'variant_generation_time': entry.get('baseline_generation_time', 0.0),
                    'variant_success': entry.get('baseline_success', False),
                    'variant_error_message': entry.get('baseline_error_message', ''),
                })

            # Update general response tracking
            has_any_response = (baseline_resp_id in response_dict) or (variant_resp_id in response_dict)
            entry.update({
                'has_model_response': has_any_response,
                'response_timestamp': datetime.now().timestamp(),
                'response_generation_timestamp': datetime.now().isoformat()
            })

            # Update stage tracking
            if 'stages_completed' not in entry:
                entry['stages_completed'] = []
            if 'responses' not in entry['stages_completed']:
                entry['stages_completed'].append('responses')
        
    
    def stage3_add_evaluation_metrics(self):
        """Stage 3: Add evaluation metrics and baseline comparisons."""
        logger.debug(f"\n🔄 Stage 3: Adding Evaluation Metrics...")
        
        # Group entries by problem_id
        problems_dict = {}
        for entry in self.data:
            # Skip entries without problem_id (malformed entries)
            if 'problem_id' not in entry:
                logger.debug(f"⚠️ Skipping entry without problem_id: {list(entry.keys())}")
                continue
            problem_id = entry['problem_id']
            if problem_id not in problems_dict:
                problems_dict[problem_id] = {'baseline': None, 'variants': []}

            if entry.get('is_baseline', False):
                problems_dict[problem_id]['baseline'] = entry
            else:
                problems_dict[problem_id]['variants'].append(entry)

        # Choose evaluation strategy based on client type
        client_type = self.config.get('eval_client_type', self.config.get('client_type', 'rits'))

        if client_type == 'vllm':
            logger.debug("🚀 Using cross-problem batched evaluation for VLLM")
            self._evaluate_all_problems_cross_batched(problems_dict)
        else:
            logger.debug("🔄 Using per-problem evaluation for RITS")
            self._evaluate_all_problems_per_problem(problems_dict)

    def _evaluate_all_problems_per_problem(self, problems_dict):
        """Original per-problem evaluation logic - used for RITS."""
        # Add evaluation metrics
        for problem_id, problem_data in problems_dict.items():
            baseline = problem_data['baseline']
            variants = problem_data['variants']

            if not baseline:
                continue

            # Get baseline response data with fallback to raw response
            baseline_answer = baseline.get('model_final_answer', '')
            if not baseline_answer:
                baseline_answer = baseline.get('model_response', '')  # Fallback to raw response

            baseline_thinking = baseline.get('model_thinking', '')
            baseline_response = baseline.get('model_response', '')
            baseline_success = baseline.get('response_success', False)
            ground_truth = baseline.get('ground_truth_answer', '')

            # Initialize answer matcher for this problem
            if not hasattr(self, '_matcher'):
                use_llm_judge = self.config.get('use_llm_judge', False)

                if use_llm_judge:
                    try:
                        logger.debug("✅ Using LLM judge for answer evaluation")
                        self._matcher = LLMAnswerMatcher(
                            client_type=self.config.get('client_type', 'rits'),
                            model_name=self.config.get('judge_model', self.config.get('response_model', 'mistral_small_3_2_instruct')),
                            use_llm=True,
                            batch_size=self.batch_size
                        )
                    except Exception as e:
                        logger.debug(f"⚠️ LLM Judge failed, using string fallback: {e}")
                        self._matcher = LLMAnswerMatcher(use_llm=False)
                else:
                    logger.debug("✅ Using string-based answer matching (default)")
                    self._matcher = LLMAnswerMatcher(use_llm=False)

            # Evaluate baseline answer against ground truth
            if ground_truth:
                baseline_eval = self._matcher.check_answer_match(
                    ground_truth=ground_truth,
                    model_answer=baseline_answer,
                    problem_context=baseline.get('original_problem', '')
                )

                # Update baseline evaluation with LLM judge results
                baseline.update({
                    'baseline_model_answer': baseline_answer,
                    'baseline_model_thinking': baseline_thinking,
                    'baseline_model_response': baseline_response,
                    'baseline_response_success': baseline_success,
                    'baseline_matches_ground_truth': baseline_eval['is_correct'],
                    'baseline_confidence': baseline_eval['confidence'],
                    'baseline_explanation': baseline_eval['explanation'],
                    'baseline_evaluation_method': baseline_eval.get('method_used', 'unknown'),
                    'variant_matches_ground_truth': baseline_eval['is_correct'],  # Same for baseline
                    'variant_confidence': baseline_eval['confidence'],
                    'variant_evaluation_method': baseline_eval.get('method_used', 'unknown'),
                    'baseline_variant_consistent': True,  # Always true for baseline
                    'positive_drift': False,  # No drift for baseline
                    'negative_drift': False,
                    'has_drift': False,
                    'has_improvement': False
                })
            else:
                # No ground truth available
                baseline.update({
                    'baseline_model_answer': baseline_answer,
                    'baseline_model_thinking': baseline_thinking,
                    'baseline_model_response': baseline_response,
                    'baseline_response_success': baseline_success,
                    'baseline_matches_ground_truth': False,
                    'variant_matches_ground_truth': False,
                    'baseline_variant_consistent': True,
                    'positive_drift': False,
                    'negative_drift': False,
                    'has_drift': False,
                    'has_improvement': False
                })

            # Update variant evaluations using batched evaluation
            self._evaluate_variants_batched(variants, baseline, baseline_answer, baseline_thinking, baseline_response, baseline_success, ground_truth)

            # Update stage tracking for all variants
            for variant in variants:
                # Update stage tracking
                if 'stages_completed' not in variant:
                    variant['stages_completed'] = []
                if 'evaluation' not in variant['stages_completed']:
                    variant['stages_completed'].append('evaluation')

            # Update baseline stage tracking
            if 'stages_completed' not in baseline:
                baseline['stages_completed'] = []
            if 'evaluation' not in baseline['stages_completed']:
                baseline['stages_completed'].append('evaluation')

        self.save_data("evaluation")
        logger.debug(f"✅ Stage 3 complete: Added LLM judge evaluation with drift analysis to unified file")

    def _evaluate_all_problems_cross_batched(self, problems_dict):
        """Cross-problem batched evaluation logic - used for VLLM maximum efficiency."""
        # Step 1: Collect ALL comparisons from ALL problems
        all_comparisons = []
        comparison_metadata = []  # Track which problem/variant each comparison belongs to

        for problem_id, problem_data in problems_dict.items():
            baseline = problem_data['baseline']
            variants = problem_data['variants']

            if not baseline:
                continue

            # Get baseline response data with fallback to raw response
            baseline_answer = baseline.get('model_final_answer', '')
            if not baseline_answer:
                baseline_answer = baseline.get('model_response', '')  # Fallback to raw response

            baseline_thinking = baseline.get('model_thinking', '')
            baseline_response = baseline.get('model_response', '')
            baseline_success = baseline.get('response_success', False)
            ground_truth = baseline.get('ground_truth_answer', '')

            # Initialize answer matcher once globally
            if not hasattr(self, '_matcher'):
                use_llm_judge = self.config.get('use_llm_judge', False)
                if use_llm_judge:
                    try:
                        logger.debug("✅ Using LLM judge for answer evaluation")
                        self._matcher = LLMAnswerMatcher(
                            client_type=self.config.get('client_type', 'rits'),
                            model_name=self.config.get('judge_model', self.config.get('response_model', 'mistral_small_3_2_instruct')),
                            use_llm=True,
                            batch_size=self.batch_size
                        )
                    except Exception as e:
                        logger.debug(f"⚠️ LLM Judge failed, using string fallback: {e}")
                        self._matcher = LLMAnswerMatcher(use_llm=False)
                else:
                    logger.debug("✅ Using string-based answer matching (default)")
                    self._matcher = LLMAnswerMatcher(use_llm=False)

            # Collect comparisons from this problem
            if ground_truth and baseline_answer:
                # Baseline comparison
                all_comparisons.append((baseline_answer, ground_truth))
                comparison_metadata.append({
                    'type': 'baseline',
                    'problem_id': problem_id,
                    'entry': baseline,
                    'baseline_answer': baseline_answer,
                    'baseline_thinking': baseline_thinking,
                    'baseline_response': baseline_response,
                    'baseline_success': baseline_success,
                    'ground_truth': ground_truth
                })

                # Variant comparisons
                for variant in variants:
                    variant_answer = variant.get('model_final_answer', '')
                    if not variant_answer:
                        variant_answer = variant.get('model_response', '')  # Fallback to raw response

                    if variant_answer:
                        all_comparisons.append((variant_answer, ground_truth))
                        comparison_metadata.append({
                            'type': 'variant',
                            'problem_id': problem_id,
                            'entry': variant,
                            'baseline_answer': baseline_answer,
                            'baseline_thinking': baseline_thinking,
                            'baseline_response': baseline_response,
                            'baseline_success': baseline_success,
                            'ground_truth': ground_truth
                        })

        if not all_comparisons:
            logger.debug("⚠️ No valid comparisons for cross-problem LLM judge evaluation")
            return

        # Step 2: Batch ALL comparisons together (cross-problem)
        client_type = self.config.get('eval_client_type', self.config.get('client_type', 'rits'))
        eval_batch_size = self.config.get('response_eval_batch_size', self.batch_size)
        if client_type == 'vllm':
            judge_batch_size = eval_batch_size
            logger.debug(f"🔄 Using VLLM eval batch size: {judge_batch_size}")
        else:
            judge_batch_size = self.config.get('rits_batch_size', 8)
            logger.debug(f"🔄 Using RITS batch size: {judge_batch_size}")

        # Check if we have LLM judge or string-based matching
        if self._matcher.judge is not None:
            logger.debug(f"🧑‍⚖️ Cross-problem batched LLM judge evaluation: {len(all_comparisons)} comparisons in batches of {judge_batch_size}")

            # Process all comparisons in batches using LLM judge
            all_results = []
            eval_range = range(0, len(all_comparisons), judge_batch_size)
            if not getattr(self, 'verbose', False):
                eval_range = tqdm(eval_range, desc='Evaluating', unit='batch')
            for i in eval_range:
                batch_comparisons = all_comparisons[i:i + judge_batch_size]
                logger.debug(f"  📦 Evaluating batch {i//judge_batch_size + 1}/{(len(all_comparisons) + judge_batch_size - 1)//judge_batch_size} ({len(batch_comparisons)} comparisons)")

                batch_results = self._matcher.judge.batch_judge_answers(batch_comparisons)
                all_results.extend(batch_results)
        else:
            logger.debug(f"🧑‍⚖️ Cross-problem string-based evaluation: {len(all_comparisons)} comparisons processed individually")

            # Process comparisons individually using string matching
            all_results = []
            for i, (model_answer, ground_truth) in enumerate(all_comparisons):
                result = self._matcher.check_answer_match(ground_truth=ground_truth, model_answer=model_answer)
                all_results.append(result['is_correct'])

        # Step 3: Distribute results back to correct problems/variants
        baseline_results_cache = {}  # Cache baseline results to avoid recomputation

        for i, (result, metadata) in enumerate(zip(all_results, comparison_metadata)):
            problem_id = metadata['problem_id']
            entry = metadata['entry']

            if metadata['type'] == 'baseline':
                # Cache baseline result for reuse
                baseline_results_cache[problem_id] = result

                # Update baseline entry
                if metadata['ground_truth']:
                    entry.update({
                        'baseline_model_answer': metadata['baseline_answer'],
                        'baseline_model_thinking': metadata['baseline_thinking'],
                        'baseline_model_response': metadata['baseline_response'],
                        'baseline_response_success': metadata['baseline_success'],
                        'baseline_matches_ground_truth': result,
                        'baseline_confidence': 'high' if result else 'medium',
                        'baseline_explanation': f'LLM judge evaluation: {result}',
                        'baseline_evaluation_method': 'llm_judge_cross_batched',
                        'variant_matches_ground_truth': result,  # Same for baseline
                        'variant_confidence': 'high' if result else 'medium',
                        'variant_evaluation_method': 'llm_judge_cross_batched',
                        'baseline_variant_consistent': True,  # Always true for baseline
                        'positive_drift': False,  # No drift for baseline
                        'negative_drift': False,
                        'has_drift': False,
                        'has_improvement': False
                    })
                else:
                    # No ground truth
                    entry.update({
                        'baseline_model_answer': metadata['baseline_answer'],
                        'baseline_model_thinking': metadata['baseline_thinking'],
                        'baseline_model_response': metadata['baseline_response'],
                        'baseline_response_success': metadata['baseline_success'],
                        'baseline_matches_ground_truth': False,
                        'variant_matches_ground_truth': False,
                        'baseline_variant_consistent': True,
                        'positive_drift': False,
                        'negative_drift': False,
                        'has_drift': False,
                        'has_improvement': False
                    })

            else:  # variant
                # Get baseline result from cache
                baseline_result = baseline_results_cache.get(problem_id, False)
                variant_result = result

                # Calculate drift
                positive_drift = variant_result and not baseline_result
                negative_drift = not variant_result and baseline_result

                # Update variant entry
                entry.update({
                    'baseline_model_answer': metadata['baseline_answer'],
                    'baseline_model_thinking': metadata['baseline_thinking'],
                    'baseline_model_response': metadata['baseline_response'],
                    'baseline_response_success': metadata['baseline_success'],

                    'baseline_matches_ground_truth': baseline_result,
                    'baseline_confidence': 'high' if baseline_result else 'medium',
                    'baseline_explanation': f'LLM judge evaluation: {baseline_result}',
                    'baseline_evaluation_method': 'llm_judge_cross_batched',

                    'variant_matches_ground_truth': variant_result,
                    'variant_confidence': 'high' if variant_result else 'medium',
                    'variant_explanation': f'LLM judge evaluation: {variant_result}',
                    'variant_evaluation_method': 'llm_judge_cross_batched',

                    'positive_drift': positive_drift,
                    'negative_drift': negative_drift,
                    'has_drift': positive_drift or negative_drift,
                    'baseline_variant_consistent': baseline_result == variant_result,
                    'has_improvement': positive_drift  # Legacy compatibility
                })

                if positive_drift:
                    logger.info(f"     📈 POSITIVE DRIFT: {entry['variation_id']}")
                elif negative_drift:
                    logger.info(f"     📉 NEGATIVE DRIFT: {entry['variation_id']}")

        # Step 4: Update stage tracking for all problems
        for problem_id, problem_data in problems_dict.items():
            baseline = problem_data['baseline']
            variants = problem_data['variants']

            if not baseline:
                continue

            # Update variant stage tracking
            for variant in variants:
                if 'stages_completed' not in variant:
                    variant['stages_completed'] = []
                if 'evaluation' not in variant['stages_completed']:
                    variant['stages_completed'].append('evaluation')

            # Update baseline stage tracking
            if 'stages_completed' not in baseline:
                baseline['stages_completed'] = []
            if 'evaluation' not in baseline['stages_completed']:
                baseline['stages_completed'].append('evaluation')

        self.save_data("evaluation")
        logger.debug(f"✅ Stage 3 complete: Added cross-problem batched LLM judge evaluation to unified file")

    def _evaluate_variants_batched(self, variants, baseline, baseline_answer, baseline_thinking, baseline_response, baseline_success, ground_truth):
        """Evaluate variants using batched LLM judge calls with client-specific batch sizes."""
        if not variants:
            return

        # Collect all comparisons that need LLM judging
        comparisons = []
        variant_indices = []

        for i, variant in enumerate(variants):
            variant_answer = variant.get('model_final_answer', '')
            if not variant_answer:
                variant_answer = variant.get('model_response', '')  # Fallback to raw response

            # Add baseline comparison data to variant
            variant['baseline_model_answer'] = baseline_answer
            variant['baseline_model_thinking'] = baseline_thinking
            variant['baseline_model_response'] = baseline_response
            variant['baseline_response_success'] = baseline_success

            if ground_truth and baseline_answer and variant_answer:
                # Collect comparisons for batching: (baseline_answer, ground_truth) and (variant_answer, ground_truth)
                comparisons.extend([
                    (baseline_answer, ground_truth),
                    (variant_answer, ground_truth)
                ])
                variant_indices.append(i)
            else:
                # Fallback evaluation for variants without sufficient data
                variant['baseline_matches_ground_truth'] = False
                variant['variant_matches_ground_truth'] = False
                variant['baseline_variant_consistent'] = False
                variant['positive_drift'] = False
                variant['negative_drift'] = False
                variant['has_drift'] = False
                variant['has_improvement'] = False

        if not comparisons:
            logger.debug("⚠️ No valid comparisons for LLM judge evaluation")
            return

        # Determine batch size based on client type
        client_type = self.config.get('eval_client_type', self.config.get('client_type', 'rits'))
        eval_batch_size = self.config.get('response_eval_batch_size', self.batch_size)
        if client_type == 'vllm':
            judge_batch_size = eval_batch_size
            logger.debug(f"🔄 Using VLLM eval batch size: {judge_batch_size}")
        else:
            judge_batch_size = self.config.get('rits_batch_size', 8)  # Default RITS batch size
            logger.debug(f"🔄 Using RITS batch size: {judge_batch_size}")

        # Check if we have LLM judge or string-based matching
        if self._matcher.judge is not None:
            logger.debug(f"🧑‍⚖️ Batched LLM judge evaluation: {len(comparisons)} comparisons in batches of {judge_batch_size}")

            # Process comparisons in batches using LLM judge
            all_results = []
            eval_range = range(0, len(comparisons), judge_batch_size)
            if not getattr(self, 'verbose', False):
                eval_range = tqdm(eval_range, desc='Evaluating', unit='batch')
            for i in eval_range:
                batch_comparisons = comparisons[i:i + judge_batch_size]
                logger.debug(f"  📦 Evaluating batch {i//judge_batch_size + 1}/{(len(comparisons) + judge_batch_size - 1)//judge_batch_size} ({len(batch_comparisons)} comparisons)")

                batch_results = self._matcher.judge.batch_judge_answers(batch_comparisons)
                all_results.extend(batch_results)
        else:
            logger.debug(f"🧑‍⚖️ String-based evaluation: {len(comparisons)} comparisons processed individually")

            # Process comparisons individually using string matching
            all_results = []
            for i, (model_answer, ground_truth) in enumerate(comparisons):
                result = self._matcher.check_answer_match(ground_truth=ground_truth, model_answer=model_answer)
                all_results.append(result['is_correct'])

        # Distribute results back to variants
        for i, variant_idx in enumerate(variant_indices):
            variant = variants[variant_idx]

            # Each variant has 2 results: baseline_result and variant_result
            result_idx = i * 2
            baseline_result = all_results[result_idx] if result_idx < len(all_results) else False
            variant_result = all_results[result_idx + 1] if result_idx + 1 < len(all_results) else False

            # Calculate drift
            positive_drift = variant_result and not baseline_result  # Variant correct, baseline wrong
            negative_drift = not variant_result and baseline_result  # Variant wrong, baseline correct

            # Update variant with results
            variant.update({
                'baseline_matches_ground_truth': baseline_result,
                'baseline_confidence': 'high' if baseline_result else 'medium',
                'baseline_explanation': f'LLM judge evaluation: {baseline_result}',
                'baseline_evaluation_method': 'llm_judge_batched',

                'variant_matches_ground_truth': variant_result,
                'variant_confidence': 'high' if variant_result else 'medium',
                'variant_explanation': f'LLM judge evaluation: {variant_result}',
                'variant_evaluation_method': 'llm_judge_batched',

                'positive_drift': positive_drift,
                'negative_drift': negative_drift,
                'has_drift': positive_drift or negative_drift,
                'baseline_variant_consistent': baseline_result == variant_result,
                'has_improvement': positive_drift  # Legacy compatibility
            })

            if positive_drift:
                logger.info(f"     📈 POSITIVE DRIFT: {variant['variation_id']}")
            elif negative_drift:
                logger.info(f"     📉 NEGATIVE DRIFT: {variant['variation_id']}")

    def export_to_csv(self, csv_path: Optional[str] = None):
        """Export unified data to CSV for analysis."""
        if not csv_path:
            csv_path = self.unified_file.replace('.json', '.csv')

        # Define column order for CSV
        priority_columns = [
            # Problem identification
            'problem_id', 'variation_id', 'variation_type', 'is_baseline', 'is_variant',
            'variant_number', 'total_variants_for_problem',

            # Problem content
            'original_problem', 'modified_problem', 'baseline_problem',

            # Ground truth and evaluation
            'ground_truth_answer', 'baseline_model_answer', 'baseline_model_thinking',
            'baseline_matches_ground_truth', 'variant_matches_ground_truth',
            'baseline_variant_consistent', 'has_drift', 'has_improvement',

            # Transformation details
            'transformation_type', 'original_component', 'new_component',
            'debugging_capability', 'generation_method', 'detection_method',

            # Model response data
            'has_model_response', 'response_success', 'model_final_answer',
            'model_thinking', 'model_response', 'response_generation_time',

            # Metadata
            'stages_completed', 'confidence', 'variation_index'
        ]

        df = pd.DataFrame(self.data)

        # Reorder columns
        existing_priority_cols = [col for col in priority_columns if col in df.columns]
        other_cols = [col for col in df.columns if col not in priority_columns]
        column_order = existing_priority_cols + sorted(other_cols)
        df = df[column_order]

        # Clean up list/dict columns for CSV
        for col in df.columns:
            if col in ['domains_involved', 'candidates_transformed', 'selected_variants', 'stages_completed']:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        df.to_csv(csv_path, index=False)

        # Print statistics
        total_rows = len(df)
        baseline_count = len(df[df['is_baseline'] == True])
        variant_count = len(df[df['is_variant'] == True])
        response_count = len(df[df['has_model_response'] == True])
        successful_responses = len(df[df['response_success'] == True])
        drift_count = len(df[df['has_drift'] == True])
        improvement_count = len(df[df['has_improvement'] == True])

        logger.debug(f"\n📊 Export Statistics:")
        logger.debug(f"   Total entries: {total_rows}")
        logger.debug(f"   Baseline problems: {baseline_count}")
        logger.debug(f"   Variant problems: {variant_count}")
        logger.debug(f"   With model responses: {response_count}")
        logger.debug(f"   Successful responses: {successful_responses}")
        logger.debug(f"   Drift cases: {drift_count}")
        logger.debug(f"   Improvement cases: {improvement_count}")
        logger.debug(f"   📄 Exported to: {csv_path}")

    def _answers_match(self, answer1: str, answer2: str) -> bool:
        """Check if two answers match using normalized comparison."""
        if not answer1 or not answer2:
            return False

        # Normalize answers for comparison
        norm1 = self._normalize_answer(answer1)
        norm2 = self._normalize_answer(answer2)

        return norm1 == norm2

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        import re

        if not answer:
            return ''

        # Convert to lowercase and strip
        normalized = answer.lower().strip()

        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(the answer is|answer:|result:|solution:)\s*', '', normalized)
        normalized = re.sub(r'\s*(dollars?|cents?|units?)\s*$', '', normalized)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove trailing punctuation
        normalized = re.sub(r'[.!?]+$', '', normalized)

        return normalized
    
    def run_complete_pipeline(self) -> Dict[str, str]:
        """Run the complete unified pipeline with all stages."""
        logger.debug(f"\n🚀 RUNNING UNIFIED PROGRESSIVE PIPELINE")
        logger.debug(f"{'='*80}")
        logger.debug(f"Unified file: {self.unified_file}")
        logger.debug(f"{'='*80}")

        results = {
            'unified_file': self.unified_file,
            'csv_file': self.unified_file.replace('.json', '.csv')
        }
        
        try:
            # Stage 1: Generate variations
            if not self.config.get('skip_variations', False):
                self.stage1_generate_variations()
            else:
                logger.debug(f"⏩ Skipping variation generation")

            # Stage 2: Generate responses
            if not self.config.get('skip_responses', False):
                self.stage2_generate_responses(force_regenerate=self.config.get('force_regenerate', False))
            else:
                logger.debug(f"⏩ Skipping response generation")

            # Stage 3: Add evaluation metrics
            if not self.config.get('skip_evaluation', False):
                self.stage3_add_evaluation_metrics()
            else:
                logger.debug(f"⏩ Skipping evaluation metrics")

            # Export to CSV
            if not self.config.get('skip_csv', False):
                self.export_to_csv(results['csv_file'])
            else:
                logger.debug(f"⏩ Skipping CSV export")

            logger.debug(f"\n🎉 UNIFIED PIPELINE FINISHED SUCCESSFULLY!")
            logger.debug(f"{'='*50}")
            logger.debug(f"📊 Generated files:")
            logger.debug(f"   📄 Unified data: {self.unified_file}")
            if not self.config.get('skip_csv', False):
                logger.debug(f"   📄 Analysis CSV: {results['csv_file']}")

            return results

        except Exception as e:
            logger.debug(f"❌ Pipeline failed: {e}")
            if self.config.get('verbose', False):
                import traceback
                traceback.print_exc()
            return results


# No longer needed - problems loaded directly in stage1


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Unified Progressive Variation Pipeline - Single File Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline from start to finish - ONE OUTPUT FILE
  python complete_variation_pipeline.py --input problems.jsonl --unified-file results.json --all-stages
  python complete_variation_pipeline.py --input problems.json --unified-file results.json --all-stages
  python complete_variation_pipeline.py --input problems.txt --unified-file results.json --all-stages
  python complete_variation_pipeline.py --input problems.csv --unified-file results.json --all-stages

  # Run individual stages (progressive enhancement of SAME FILE)
  python complete_variation_pipeline.py --unified-file results.json --stage variations --input problems.jsonl
  python complete_variation_pipeline.py --unified-file results.json --stage responses
  python complete_variation_pipeline.py --unified-file results.json --stage evaluation
  python complete_variation_pipeline.py --unified-file results.json --stage export-csv

  # Resume from existing unified file
  python complete_variation_pipeline.py --unified-file existing_results.json --stage responses
  python complete_variation_pipeline.py --unified-file existing_results.json --stage evaluation

  # Force regenerate responses (useful for debugging or trying different models)
  python complete_variation_pipeline.py --unified-file existing_results.json --stage responses --force-regenerate
  python complete_variation_pipeline.py --unified-file existing_results.json --stage responses --eval-model phi-4 --force-regenerate

  # Batched processing for large datasets
  python complete_variation_pipeline.py --input 10k_problems.jsonl --unified-file large_results.json --all-stages --batch-size 100 --max-workers 8

  # Different models and configurations
  python complete_variation_pipeline.py --input problems.jsonl --unified-file phi4_test.json --all-stages --eval-model phi-4 --client-type rits
  python complete_variation_pipeline.py --input problems.jsonl --unified-file granite_test.json --all-stages --eval-model granite-3-1-8b --num-variations 5

  # Generate variations with one model, evaluate with another
  python complete_variation_pipeline.py --input problems.jsonl --unified-file mixed_test.json --all-stages --model-name mistral_small_3_2_instruct --eval-model granite-3-1-8b
        """
    )
    
    # Required
    parser.add_argument('--unified-file', required=True,
                       help='Unified JSON file (created/updated progressively) - SINGLE OUTPUT FILE')

    # Stage control
    parser.add_argument('--stage', choices=['variations', 'responses', 'evaluation', 'export-csv'],
                       help='Run specific stage only')
    parser.add_argument('--all-stages', action='store_true',
                       help='Run all stages in sequence')

    # Input (required for variations stage) - ANY FORMAT
    parser.add_argument('--input',
                       help='Input problems file (ANY format: .json/.jsonl/.txt/.csv) - or use "test" for hardcoded example')
    
    # Configuration
    parser.add_argument('--client-type', default='rits', choices=['rits', 'openai'],
                       help='Model client type (default: rits)')
    parser.add_argument('--model-name', default='mistral_small_3_2_instruct',
                       help='Model for variation generation (default: mistral_small_3_2_instruct)')
    parser.add_argument('--eval-model', default='mistral_small_3_2_instruct',
                       help='Model being evaluated for robustness (gets responses for variations)')
    parser.add_argument('--response-model', default=None,
                       help='DEPRECATED: Use --eval-model instead')
    parser.add_argument('--use-llm-judge', action='store_true',
                       help='Use LLM judge for answer evaluation (default: string matching)')
    parser.add_argument('--judge-model', default=None,
                       help='Model to use as LLM judge (default: same as eval-model)')
    parser.add_argument('--num-variations', type=int, default=3,
                       help='Number of variations per problem (default: 3)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Max workers for parallel processing (default: 4)')
    parser.add_argument('--use-optimized', action='store_true',
                       help='Use optimized LLM-guided selector')

    # Batching configuration
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size for processing (0 = no batching, default: 0)')
    parser.add_argument('--save-every-batch', action='store_true', default=True,
                       help='Save after every batch (default: True)')

    # Control flags
    parser.add_argument('--skip-variations', action='store_true',
                       help='Skip variation generation')
    parser.add_argument('--skip-responses', action='store_true',
                       help='Skip response generation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation metrics')
    parser.add_argument('--skip-csv', action='store_true',
                       help='Skip CSV export')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate responses even if already completed')

    # Output
    parser.add_argument('--csv-output',
                       help='CSV output path (default: unified_file.csv)')

    # Other options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validation
    if args.all_stages and not args.input:
        logger.debug("🧪 No --input specified, will use hardcoded test example")
        args.input = 'test'  # Set to trigger test mode

    if args.stage == 'variations' and not args.input:
        logger.debug("🧪 No --input specified, will use hardcoded test example")
        args.input = 'test'  # Set to trigger test mode
    
    # Handle backward compatibility for response_model
    eval_model = args.eval_model
    if args.response_model is not None:
        logger.debug("⚠️ --response-model is deprecated, use --eval-model instead")
        eval_model = args.response_model

    # Configuration
    config = {
        'unified_file': args.unified_file,
        'input_problems': args.input,
        'client_type': args.client_type,
        'model_name': args.model_name,
        'response_model': eval_model,  # Model being evaluated
        'num_variations': args.num_variations,
        'max_workers': args.max_workers,
        'use_optimized': args.use_optimized,
        'use_llm_judge': args.use_llm_judge,
        'judge_model': args.judge_model,
        'use_model_client': True,
        'variation_config': {},
        'batch_size': args.batch_size,
        'save_every_batch': args.save_every_batch,
        'skip_variations': args.skip_variations,
        'skip_responses': args.skip_responses,
        'skip_evaluation': args.skip_evaluation,
        'skip_csv': args.skip_csv,
        'force_regenerate': args.force_regenerate,
        'verbose': args.verbose
    }
    
    # Initialize pipeline
    pipeline = UnifiedProgressivePipeline(config)

    logger.debug(f"🚀 Unified Pipeline Starting...")
    logger.debug(f"   Unified file: {args.unified_file}")
    if args.input:
        logger.debug(f"   Input: {args.input}")
    logger.debug(f"   Variation model: {args.model_name}")
    logger.debug(f"   Evaluation model: {eval_model}")

    try:
        if args.all_stages:
            # Run all stages
            results = pipeline.run_complete_pipeline()

        elif args.stage == 'variations':
            pipeline.stage1_generate_variations()

        elif args.stage == 'responses':
            pipeline.stage2_generate_responses(eval_model, force_regenerate=args.force_regenerate)

        elif args.stage == 'evaluation':
            pipeline.stage3_add_evaluation_metrics()

        elif args.stage == 'export-csv':
            pipeline.export_to_csv(args.csv_output)

        else:
            logger.debug("❌ Please specify --stage or --all-stages")
            return 1

        logger.debug(f"\n🎉 Pipeline complete!")
        return 0

    except Exception as e:
        logger.debug(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()