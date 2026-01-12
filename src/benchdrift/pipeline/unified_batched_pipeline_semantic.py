#!/usr/bin/env python3
"""
Unified Batched Pipeline with Semantic Clustering

SEMANTIC VERSION: Identical to unified_batched_pipeline.py with semantic clustering added to Stage 0.

Stage 0 Enhancement:
- After candidate detection (atomics + composites + fragments)
- Runs semantic clustering using embeddings + hierarchical clustering
- Optional: CAGrad dependency testing to find cluster interactions

Key Features:
- Semantic clustering: ALWAYS enabled (groups candidates by embedding similarity)
- CAGrad dependencies: OPTIONAL flag (tests which clusters interact)
- Fixed batching: No manual chunking, proper RITSClient usage
- All other stages: IDENTICAL to unified_batched_pipeline.py

Output Structure:
- candidates: List of detected candidates
- semantic_clusters: List of clusters with members (text + span for each)
- cagrad_rankings: Optional list of top-k clusters ranked by gradient score (if CAGrad enabled)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
import pandas as pd
from datetime import datetime
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

from multiprocessing import Pool
# mp.set_start_method('spawn', force=True)

# Global constants - single source of truth for defaults

DEFAULT_MAX_COMBINATIONS = 50

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import from current directory's complete_variation_pipeline (uses batched engine)
from benchdrift.pipeline.complete_variation_pipeline import UnifiedProgressivePipeline


class UnifiedBatchedPipeline(UnifiedProgressivePipeline):
    """Batched version of the unified progressive enhancer."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration including batch settings."""
        super().__init__(config)
        self.batch_size = config.get('batch_size', 50)
        self.save_every_batch = config.get('save_every_batch', True)
        self.max_combinations = config.get('max_combinations', DEFAULT_MAX_COMBINATIONS)
        self.rectify_invalid = config.get('rectify_invalid', False)  # Default: drop invalid variations

        # VLLM singleton: Create shared model client to avoid multiple GPU initializations
        self._shared_vllm_client = None
        self._client_type = config.get('client_type', 'rits')
        self._model_name = config.get('model_name')

        # Semantic clustering configuration
        self.use_cagrad_dependencies = config.get('use_cagrad_dependencies', False)  # Optional CAGrad after clustering

        # Logging configuration
        self.verbose = config.get('verbose', False)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging: file (all) + console (info only unless verbose)"""
        logger = logging.getLogger('BenchDrift')
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # Clear existing handlers
        logger.propagate = False  # Don't propagate to root logger (important for Jupyter)

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # File handler - everything
        fh = logging.FileHandler('logs/pipeline_debug.log', mode='w')
        fh.setLevel(logging.DEBUG)

        # Console handler - info or debug based on verbose
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _get_model_client_for_stage(self, stage: str):
        """Get model client for specific stage with appropriate model."""
        # Determine which model to use based on stage
        if stage == 'variations':
            model_name = self.config.get('model_name', 'mistral_small_3_2_instruct')
        elif stage == 'validation':
            # Use judge model for validation (same as evaluation)
            model_name = self.config.get('judge_model') or self.config.get('model_name', 'mistral_small_3_2_instruct')
        elif stage == 'responses':
            model_name = self.config.get('response_model', 'mistral_small_3_2_instruct')
        elif stage == 'evaluation':
            model_name = self.config.get('judge_model') or self.config.get('model_name', 'mistral_small_3_2_instruct')
        else:
            model_name = self.config.get('model_name', 'mistral_small_3_2_instruct')

        # Get max_model_len and max_new_tokens from config
        max_model_len = self.config.get('max_model_len', 8192)
        max_new_tokens = self.config.get('max_new_tokens', 1000)

        from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations
        self.logger.debug(f"🎯 Creating model client for {stage}: {model_name} with {self._client_type}")
        self.logger.debug(f"   max_model_len: {max_model_len}, max_new_tokens: {max_new_tokens}")

        # Store max_new_tokens in model_config for create_model_client_for_variations to use
        from benchdrift.models.model_config_manager import ModelConfigManager
        model_config = ModelConfigManager()
        client_settings = model_config.get_client_settings(self._client_type)
        client_settings['max_new_tokens'] = max_new_tokens

        return create_model_client_for_variations(self._client_type, model_name, max_model_len)

    def stage1_generate_variations_batched(self):
        """Stage 1: Generate variations in batches."""
        self.logger.debug(f"\n🔄 Stage 1: Generating Variations (Batched - {self.batch_size} per batch)...")
        stage_start_time = time.time()

        # Simple logic: Only use pre-detected candidates for explicit --stages-1-4 mode
        # For --all-stages (RITS), ALWAYS load from input and run fresh detection
        if hasattr(self.config, 'get') and self.config.get('execution_mode') == 'stages_1_4':
            # Only for --stages-1-4: Load pre-detected candidates from unified file
            if os.path.exists(self.unified_file):
                with open(self.unified_file, 'r') as f:
                    unified_data = json.load(f)
                if unified_data and any(entry.get('candidate_detection_complete') for entry in unified_data):
                    problems = unified_data
                    self.logger.debug(f"📂 Loaded {len(problems)} problems with pre-detected candidates from {self.unified_file}")
                    self._candidates_pre_detected = True
                else:
                    problems = self._load_input_problems()
                    self._candidates_pre_detected = False
            else:
                problems = self._load_input_problems()
                self._candidates_pre_detected = False
        else:
            # For --all-stages or any other mode: ALWAYS load from input, ALWAYS fresh detection
            problems = self._load_input_problems()
            self._candidates_pre_detected = False

        if not hasattr(self, '_candidates_pre_detected'):
            self.logger.debug(f"📂 Loaded {len(problems)} input problems")

        # Limit problems if max_problems is specified
        max_problems = self.config.get('max_problems')
        if max_problems and max_problems < len(problems):
            problems = problems[:max_problems]
            self.logger.debug(f"🔢 Limited to {max_problems} problems (--max-problems {max_problems})")

        # Process in batches
        num_batches = math.ceil(len(problems) / self.batch_size)

        # Progress bar for batches
        batch_iterator = range(num_batches)
        if not self.verbose:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", unit="batch")

        for batch_idx in batch_iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(problems))
            batch_problems = problems[start_idx:end_idx]

            self.logger.debug(f"\n📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_problems)} problems)")

            # Time this batch
            batch_start_time = time.time()

            # Process this batch
            self._process_variation_batch(batch_problems, start_idx)

            # Batch timing
            batch_duration = time.time() - batch_start_time
            self.logger.debug(f"⏱️  Batch {batch_idx + 1} completed in {batch_duration:.2f}s ({batch_duration/len(batch_problems):.2f}s per problem)")

            # Save after each batch if configured
            if self.save_every_batch:
                self.save_data(f"variations_batch_{batch_idx + 1}")
                self.logger.debug(f"💾 Saved after batch {batch_idx + 1}")

        # Final save
        self.save_data("variations")

        # Stage timing summary
        stage_duration = time.time() - stage_start_time
        stage_minutes = int(stage_duration // 60)
        stage_seconds = stage_duration % 60
        self.logger.info(f"✅ Stage 1 complete: Generated {len(self.data)} total entries across {num_batches} batches")
        self.logger.debug(f"⏱️  Stage 1 timing: {stage_minutes}m {stage_seconds:.2f}s")

    def _load_input_problems(self):
        """Load problems from input file (original logic)."""
        input_file = self.config.get('input_problems')
        if not input_file or input_file == 'test':
            # Use hardcoded test example
            problems = [{"problem": "What is 15 + 25?", "answer": "40"}]
            self.logger.debug(f"🧪 Using test example: {problems[0]['problem']}")
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
            self.logger.debug(f"📂 Loaded {len(problems)} input problems")
        return problems

    def stage_validation(self):
        """
        STANDALONE VALIDATION STAGE

        Validates ALL variation types (generic, combinations, direct, etc.) after they have been generated.
        This is a completely independent stage that:
        - Loads ALL variations from the unified file (regardless of generation method)
        - Uses the judge model to validate each variation preserves the original answer
        - Either rectifies invalid variations OR drops them (based on --rectify-invalid flag)
        - Saves validated results back to unified file

        Usage: python unified_batched_pipeline.py --stage validation --unified-file results.json
        """
        self.logger.debug(f"\n🔄 VALIDATION STAGE: Validating All Variations...")
        self.logger.debug(f"   This stage validates ALL variation types: generic, combinations, direct, etc.")
        stage_start_time = time.time()

        # Load existing data from unified file (completely standalone - no dependency on stage 1)
        if not os.path.exists(self.unified_file):
            self.logger.debug(f"❌ Error: Unified file {self.unified_file} not found. Run stage 'variations' first.")
            return

        with open(self.unified_file, 'r') as f:
            self.data = json.load(f)
        self.logger.debug(f"📂 Loaded {len(self.data)} existing entries from {self.unified_file}")

        # Get model client for validation (uses judge model - completely independent client)
        model_client = self._get_model_client_for_stage('validation')
        if not model_client:
            self.logger.debug(f"❌ Failed to create model client for validation")
            return

        # Create variation engine with validation model
        # Note: We need UnifiedVariationEngine which has the batch validation methods
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
        engine = UnifiedVariationEngine(
            model_client=model_client,
            judge_model_client=model_client  # Use same client for validation
        )

        # Group entries by baseline problem for validation
        # NOTE: Collects ALL variations regardless of type (generic, combination, direct, etc.)
        baseline_problems = {}
        for entry in self.data:
            if entry.get('is_baseline'):
                problem_id = entry.get('problem_id', entry.get('id', ''))
                baseline_problems[problem_id] = {
                    'problem_text': entry.get('original_problem', entry.get('problem', '')),
                    'ground_truth': entry.get('ground_truth_answer', entry.get('answer', '')),
                    'variations': []
                }

        # Collect variations for each baseline
        for entry in self.data:
            if entry.get('is_variant'):
                problem_id = entry.get('problem_id', '')
                if problem_id in baseline_problems:
                    # Append the entire entry dict (not just the string) so validation can access fields
                    baseline_problems[problem_id]['variations'].append(entry)

        self.logger.debug(f"📊 Found {len(baseline_problems)} baseline problems with variations")

        # Prepare data for validation
        problems_with_variations = []
        problem_id_map = []
        for problem_id, problem_info in baseline_problems.items():
            if problem_info['variations']:
                problems_with_variations.append((problem_info['problem_text'], problem_info['variations']))
                problem_id_map.append(problem_id)

        if not problems_with_variations:
            self.logger.debug("⚠️ No variations found to validate")
            return

        self.logger.debug(f"🔍 Validating {sum(len(v[1]) for v in problems_with_variations)} variations across {len(problems_with_variations)} problems...")

        # Step 1: Global validation (batched across all problems)
        # Add progress bar for validation
        validation_iterator = tqdm(range(1), desc='Validating variations', disable=self.verbose) if not self.verbose else range(1)
        for _ in validation_iterator:
            invalid_by_problem = engine._batch_validate_all_variations(problems_with_variations, batch_size=self.batch_size)

        # Step 2: Handle invalid variations based on rectify_invalid flag
        if invalid_by_problem:
            if self.rectify_invalid:
                # Rectify invalid variations
                self.logger.debug(f"        🔧 Rectifying {sum(len(v) for v in invalid_by_problem.values())} invalid variations...")
                problems_with_invalid = []

                for problem_map_idx, problem_id in enumerate(problem_id_map):
                    if problem_map_idx not in invalid_by_problem:
                        continue

                    problem_info = baseline_problems[problem_id]
                    invalid_indices = invalid_by_problem[problem_map_idx]

                    problems_with_invalid.append((
                        problem_map_idx,  # Store problem_map_idx (numeric index, not string problem_id)
                        problem_info['problem_text'],
                        problem_info['ground_truth'],
                        problem_info['variations'],
                        invalid_indices
                    ))

                # Batch rectification: ONE API call for all problems
                corrected_by_problem_idx = engine._batch_rectify_across_problems(problems_with_invalid, batch_size=self.batch_size)

                # Update entries with corrected variations
                for problem_idx, corrected_list in corrected_by_problem_idx.items():
                    problem_map_idx = problems_with_invalid[problem_idx][0]  # Get problem_map_idx from stored tuple
                    problem_id = problem_id_map[problem_map_idx]  # Map to actual problem_id

                    # Update variation entries in self.data
                    variant_count = 0
                    for entry in self.data:
                        if entry.get('is_variant') and entry.get('problem_id') == problem_id:
                            # Check if this variant index needs correction
                            for inv_idx, corr_var in corrected_list:
                                if variant_count == inv_idx:
                                    entry['modified_problem'] = corr_var
                                    entry['validation_corrected'] = True
                                    break
                            variant_count += 1

                self.logger.debug(f"        ✅ Rectified {sum(len(v) for v in corrected_by_problem_idx.values())} invalid variations")
            else:
                # Drop invalid variations
                total_invalid = sum(len(v) for v in invalid_by_problem.values())
                self.logger.debug(f"        🗑️  Dropping {total_invalid} invalid variations...")

                entries_to_remove = []
                for problem_map_idx, problem_id in enumerate(problem_id_map):
                    if problem_map_idx not in invalid_by_problem:
                        continue

                    invalid_indices = sorted(invalid_by_problem[problem_map_idx], reverse=True)

                    # Find and mark entries to remove
                    variant_count = 0
                    for idx, entry in enumerate(self.data):
                        if entry.get('is_variant') and entry.get('problem_id') == problem_id:
                            if variant_count in invalid_indices:
                                entries_to_remove.append(idx)
                            variant_count += 1

                # Remove invalid entries (in reverse order to maintain indices)
                for idx in sorted(entries_to_remove, reverse=True):
                    del self.data[idx]

                self.logger.debug(f"        ✅ Dropped {len(entries_to_remove)} invalid variation entries")
        else:
            self.logger.debug("✅ All variations are valid")

        # Mark validation as complete
        for entry in self.data:
            entry['validation_complete'] = True

        # Save updated data
        self.save_data("validation")

        # Stage timing summary
        stage_duration = time.time() - stage_start_time
        stage_minutes = int(stage_duration // 60)
        stage_seconds = stage_duration % 60
        remaining_variations = sum(1 for entry in self.data if entry.get('is_variant'))
        self.logger.debug(f"✅ VALIDATION STAGE complete: {remaining_variations} validated variations remaining")
        self.logger.debug(f"⏱️  Validation timing: {stage_minutes}m {stage_seconds:.2f}s")

    def stage0_candidate_detection_only(self):
        """Stage 0: Candidate detection only (spaCy-based, no VLLM calls)."""
        self.logger.debug(f"\n🔍 Stage 0: Candidate Detection (CPU-only, no GPU conflicts)...")
        stage_start_time = time.time()

        # Load input problems (same logic as stage 1)
        input_file = self.config.get('input_problems')
        if not input_file or input_file == 'test':
            problems = [{"problem": "What is 15 + 25?", "answer": "40"}]
            self.logger.debug(f"🧪 Using test example: {problems[0]['problem']}")
        else:
            with open(input_file, 'r') as f:
                if input_file.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, dict):
                        problems = [data]
                    elif isinstance(data, list):
                        problems = data
                    else:
                        problems = [{"problem": str(data)}]
                else:  # jsonl
                    problems = [json.loads(line) for line in f if line.strip()]

        self.logger.debug(f"📂 Loaded {len(problems)} input problems")

        # Limit problems if specified
        max_problems = self.config.get('max_problems')
        if max_problems and max_problems < len(problems):
            problems = problems[:max_problems]
            self.logger.debug(f"🔢 Limited to {max_problems} problems (--max-problems {max_problems})")

        # Process in batches - CANDIDATE DETECTION ONLY
        num_batches = math.ceil(len(problems) / self.batch_size)
        max_workers = self.config.get('max_workers', 4)

        self.logger.debug(f"🚀 Using {max_workers} parallel workers for candidate detection")
        self.logger.debug(f"   Each worker uses spaCy batching (batch_size=50) for optimal performance")

        # Prepare batch info for all batches
        batch_infos = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(problems))
            batch_problems = problems[start_idx:end_idx]
            batch_infos.append((batch_idx, batch_problems, start_idx))

        # Process batches in parallel using ThreadPoolExecutor
        # (ThreadPoolExecutor is used instead of multiprocessing because spaCy models can't be pickled)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_single_batch_worker, batch_info): batch_info[0]
                for batch_info in batch_infos
            }

            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    self.logger.debug(f"\n✅ Batch {batch_idx + 1}/{num_batches} completed")

                    # Save batch if configured
                    if self.save_every_batch:
                        self.save_data('candidate_detection')
                        self.logger.debug(f"💾 Saved batch {batch_idx + 1} candidates to {self.unified_file}")

                except Exception as exc:
                    self.logger.debug(f"\n❌ Batch {batch_idx + 1} generated an exception: {exc}")

        # Final save if not saving every batch
        if not self.save_every_batch:
            self.save_data('candidate_detection')

        stage_duration = time.time() - stage_start_time
        stage_minutes = int(stage_duration // 60)
        stage_seconds = stage_duration % 60
        self.logger.info(f"✅ Stage 0 complete: Detected candidates for {len(problems)} problems")
        self.logger.debug(f"⏱️  Stage 0 timing: {stage_minutes}m {stage_seconds:.2f}s")

    def _detect_candidates_only_batch(self, batch_problems: List[Dict], start_idx: int):
        """
        Detect candidates for a batch using BATCHED detection with parallel workers.
        Combines spaCy batching (batch_size=50) with parallel processing across workers.
        """

        # Extract problem texts and metadata
        problem_texts = []
        problem_ids = []
        ground_truths = []

        for idx, problem_data in enumerate(batch_problems):
            if isinstance(problem_data, dict):
                problem_text = problem_data.get('problem', problem_data.get('question', str(problem_data)))
                problem_id = problem_data.get('id', f'problem_{start_idx + idx}')
                ground_truth = problem_data.get('answer', problem_data.get('ground_truth', ''))
            else:
                problem_text = str(problem_data)
                problem_id = f'problem_{start_idx + idx}'
                ground_truth = ''

            problem_texts.append(problem_text)
            problem_ids.append(problem_id)
            ground_truths.append(ground_truth)

        # Create engine ONCE for entire batch
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
        engine = UnifiedVariationEngine(model_client=None)

        # ✅ BATCH DETECTION: Process all problems at once (10-50x faster than sequential)
        self.logger.debug(f"  🚀 Batch detecting candidates for {len(problem_texts)} problems...")
        try:
            all_candidates = engine._detect_all_candidates_batch(problem_texts)

            # Apply dependency detection to each problem's candidates
            for i, candidates in enumerate(all_candidates):
                if candidates:
                    all_candidates[i] = engine._detect_and_merge_dependencies(problem_texts[i], candidates)

            # ═══════════════════════════════════════════════════════════════════
            # ADD FRAGMENTS: Detect fragments and add to candidate list
            # ═══════════════════════════════════════════════════════════════════
            from benchdrift.pipeline.comprehensive_variation_engine_v2 import detect_fragments_comprehensive

            for i, problem_text in enumerate(problem_texts):
                # Detect fragments
                fragments = detect_fragments_comprehensive(problem_text)

                # Convert fragments to candidate format and add to list
                if fragments:
                    existing_texts = {cand['text'] for cand in all_candidates[i]}
                    fragments_added = 0

                    for frag in fragments:
                        # Deduplicate: only add if not already in candidates
                        if frag.text not in existing_texts:
                            all_candidates[i].append({
                                'text': frag.text,
                                'domain': 'fragment',  # Mark as fragment
                                'topic': frag.type,     # Use fragment type as topic
                                'pos': frag.span[0],
                                'fragment_priority': frag.priority,
                                'fragment_type': frag.type
                            })
                            fragments_added += 1

                    if fragments_added > 0:
                        self.logger.debug(f"      📦 Added {fragments_added} fragments as candidates (deduplicated)")

            # ═══════════════════════════════════════════════════════════════════
            # SEMANTIC CLUSTERING: Cluster candidates by embedding similarity
            # ═══════════════════════════════════════════════════════════════════
            from benchdrift.pipeline.semantic_composite_detector import SemanticCompositeDetector, Candidate

            # Initialize semantic detector once for the batch
            semantic_detector = SemanticCompositeDetector(
                embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2'),
                semantic_threshold=self.config.get('semantic_threshold', 0.35),
                verbose=self.verbose
            )

            all_clusters = []
            all_dependency_graphs = []

            for i, (problem_text, candidates) in enumerate(zip(problem_texts, all_candidates)):
                if not candidates:
                    all_clusters.append([])
                    all_dependency_graphs.append(None)
                    continue

                # Convert candidate dicts to Candidate objects
                candidate_objects = []
                for cand in candidates:
                    # Get span - handle both 'pos' and 'span' fields
                    if 'span' in cand:
                        span = cand['span']
                    elif 'pos' in cand:
                        # pos is just start position, estimate end
                        start = cand['pos']
                        end = start + len(cand['text'])
                        span = (start, end)
                    else:
                        # No position info - skip this candidate
                        continue

                    candidate_objects.append(Candidate(
                        text=cand['text'],
                        span=span,
                        type=cand.get('topic', cand.get('domain', 'unknown'))
                    ))

                # Run semantic clustering (ALWAYS)
                self.logger.debug(f"  🔬 Running semantic clustering for problem {i+1}/{len(problem_texts)}...")
                clusters = semantic_detector.detect_clusters(problem_text, candidate_objects)
                all_clusters.append(clusters)
                self.logger.debug(f"     ✅ Found {len(clusters)} semantic clusters")

                # Optional: CAGrad cluster ranking (CONDITIONAL)
                if self.use_cagrad_dependencies and clusters:
                    self.logger.debug(f"  🧪 Running CAGrad cluster ranking for problem {i+1}/{len(problem_texts)}...")
                    top_k_clusters = self._run_cagrad_batched(
                        problem_text,
                        ground_truths[i],
                        clusters,
                        semantic_detector.embedder
                    )
                    all_dependency_graphs.append(top_k_clusters)  # Reuse variable name for compatibility
                else:
                    all_dependency_graphs.append(None)

            # Store results with semantic clusters and dependency graphs
            for i, (problem_id, problem_text, ground_truth, candidates, clusters, dep_graph) in enumerate(
                zip(problem_ids, problem_texts, ground_truths, all_candidates, all_clusters, all_dependency_graphs)):

                # Serialize clusters with full span info for each member
                clusters_data = []
                for cluster in clusters:
                    clusters_data.append({
                        'cluster_id': cluster.cluster_id,
                        'members': [
                            {
                                'text': m.text,
                                'span': list(m.span),  # (start, end) as list for JSON
                                'type': m.type
                            } for m in cluster.members
                        ]
                    })

                # Serialize CAGrad cluster rankings if present
                cagrad_rankings_data = None
                if dep_graph is not None:
                    # dep_graph is actually top_k_clusters: list of (cluster_id, gradient) tuples
                    cagrad_rankings_data = [
                        {'cluster_id': cluster_id, 'gradient': gradient}
                        for cluster_id, gradient in dep_graph
                    ]

                self.data.append({
                    'id': problem_id,
                    'problem': problem_text,
                    'answer': ground_truth,
                    'candidates': candidates,
                    'semantic_clusters': clusters_data,  # Clusters with spans for each keyword
                    'cagrad_rankings': cagrad_rankings_data,  # CAGrad top-k cluster rankings (optional)
                    'candidate_detection_complete': True,
                    'is_baseline': True,  # This is the original problem
                    'is_variant': False,  # Not a variation
                    'timestamp': datetime.now().isoformat()
                })

                if clusters:
                    cagrad_str = f", top-{len(dep_graph)} CAGrad clusters" if dep_graph else ""
                    self.logger.debug(f"  🔍 Problem {problem_id}: {problem_text[:60]}... ✅ {len(candidates)} candidates → {len(clusters)} clusters{cagrad_str}")
                else:
                    self.logger.debug(f"  🔍 Problem {problem_id}: {problem_text[:60]}... ✅ {len(candidates)} candidates")

        except Exception as e:
            # Fallback: Process individually if batch fails (same logic as old multiprocessing error handling)
            self.logger.debug(f"  ⚠️ Batch detection failed: {e}. Falling back to individual processing...")
            from benchdrift.pipeline.comprehensive_variation_engine_v2 import detect_fragments_comprehensive

            for i, (problem_id, problem_text, ground_truth) in enumerate(zip(problem_ids, problem_texts, ground_truths)):
                try:
                    candidates = engine._detect_all_candidates_with_composites(problem_text)
                    if candidates:
                        candidates = engine._detect_and_merge_dependencies(problem_text, candidates)

                    # Add fragments (same as batch path)
                    fragments = detect_fragments_comprehensive(problem_text)
                    if fragments:
                        existing_texts = {cand['text'] for cand in candidates}
                        for frag in fragments:
                            if frag.text not in existing_texts:
                                candidates.append({
                                    'text': frag.text,
                                    'domain': 'fragment',
                                    'topic': frag.type,
                                    'pos': frag.span[0],
                                    'fragment_priority': frag.priority,
                                    'fragment_type': frag.type
                                })

                    # Semantic clustering (fallback path)
                    from benchdrift.pipeline.semantic_composite_detector import SemanticCompositeDetector, Candidate
                    semantic_detector = SemanticCompositeDetector(
                        embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2'),
                        semantic_threshold=self.config.get('semantic_threshold', 0.35)
                    )

                    # Convert to Candidate objects
                    candidate_objects = []
                    for cand in candidates:
                        if 'span' in cand:
                            span = cand['span']
                        elif 'pos' in cand:
                            start = cand['pos']
                            end = start + len(cand['text'])
                            span = (start, end)
                        else:
                            continue
                        candidate_objects.append(Candidate(
                            text=cand['text'],
                            span=span,
                            type=cand.get('topic', cand.get('domain', 'unknown'))
                        ))

                    # Cluster (ALWAYS)
                    clusters = semantic_detector.detect_clusters(problem_text, candidate_objects) if candidate_objects else []

                    # Optional CAGrad cluster ranking (CONDITIONAL)
                    top_k_clusters = None
                    if self.use_cagrad_dependencies and clusters:
                        top_k_clusters = self._run_cagrad_batched(problem_text, ground_truth, clusters, semantic_detector.embedder)

                    # Serialize clusters
                    clusters_data = []
                    for cluster in clusters:
                        clusters_data.append({
                            'cluster_id': cluster.cluster_id,
                            'members': [
                                {'text': m.text, 'span': list(m.span), 'type': m.type}
                                for m in cluster.members
                            ]
                        })

                    # Serialize CAGrad cluster rankings
                    cagrad_rankings_data = None
                    if top_k_clusters is not None:
                        cagrad_rankings_data = [
                            {'cluster_id': cluster_id, 'gradient': gradient}
                            for cluster_id, gradient in top_k_clusters
                        ]

                    self.data.append({
                        'id': problem_id,
                        'problem': problem_text,
                        'answer': ground_truth,
                        'candidates': candidates,
                        'semantic_clusters': clusters_data,
                        'cagrad_rankings': cagrad_rankings_data,
                        'candidate_detection_complete': True,
                        'is_baseline': True,  # This is the original problem
                        'is_variant': False,  # Not a variation
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as inner_e:
                    # Individual problem error - same error handling as before
                    self.data.append({
                        'id': problem_id,
                        'problem': problem_text,
                        'answer': ground_truth,
                        'candidates': [],
                        'candidate_detection_complete': True,
                        'detection_error': str(inner_e),
                        'timestamp': datetime.now().isoformat()
                    })

    def _process_single_batch_worker(self, batch_info):
        """Worker function to process a single batch (for parallel processing)."""
        batch_idx, batch_problems, start_idx = batch_info

        # Call the existing batch detection logic (unchanged)
        self._detect_candidates_only_batch(batch_problems, start_idx)

        return batch_idx

    @staticmethod
    def _detect_candidates_for_single_problem(args):
        """
        DEPRECATED: This method is no longer used (replaced by _detect_all_candidates_batch).
        Kept for backward compatibility only.

        Old helper function for parallel candidate detection using multiprocessing.
        Now replaced with spaCy batching which is 10-50x faster.
        """
        problem_data, actual_idx = args
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine

        # Initialize engine (no model client needed)
        engine = UnifiedVariationEngine(model_client=None)

        # Extract problem data
        if isinstance(problem_data, dict):
            problem_text = problem_data.get('problem', problem_data.get('question', str(problem_data)))
            problem_id = problem_data.get('id', f'problem_{actual_idx}')
            ground_truth = problem_data.get('answer', problem_data.get('ground_truth', ''))
        else:
            problem_text = str(problem_data)
            problem_id = f'problem_{actual_idx}'
            ground_truth = ''

        if not problem_text:
            return None

        try:
            # EXACT same candidate detection as main pipeline
            candidates = engine._detect_all_candidates_with_composites(problem_text)

            # Apply dependency detection exactly like main pipeline
            if candidates:
                candidates = engine._detect_and_merge_dependencies(problem_text, candidates)

            # Return result entry
            return {
                'id': problem_id,
                'problem': problem_text,
                'answer': ground_truth,
                'candidates': candidates,
                'candidate_detection_complete': True,
                'timestamp': datetime.now().isoformat(),
                'num_candidates': len(candidates)
            }
        except Exception as e:
            # Return entry with error
            return {
                'id': problem_id,
                'problem': problem_text,
                'answer': ground_truth,
                'candidates': [],
                'candidate_detection_complete': True,
                'detection_error': str(e),
                'timestamp': datetime.now().isoformat(),
                'num_candidates': 0
            }

    @staticmethod
    def _detect_candidates_for_stage1(args):
        """
        DEPRECATED: This method is no longer used (replaced by _detect_all_candidates_batch).
        Kept for backward compatibility only.

        Old helper for parallel candidate detection in Stage 1 using multiprocessing.
        Now replaced with spaCy batching which is 10-50x faster.
        """
        problem_info, problem = args
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine

        # Initialize engine (no model client needed for detection)
        engine = UnifiedVariationEngine(model_client=None)

        actual_idx = problem_info['actual_idx']
        problem_text = problem_info['problem_text']

        # Check if using pre-detected candidates
        if isinstance(problem, dict) and problem.get('candidates'):
            # Return pre-detected candidates as-is (will be converted later)
            return {
                'idx': problem_info['idx'],
                'actual_idx': actual_idx,
                'candidates': problem.get('candidates', []),
                'pre_detected': True,
                'num_candidates': len(problem.get('candidates', []))
            }
        else:
            # Normal detection path
            candidates = engine._detect_all_candidates_with_composites(problem_text)
            # Apply dependency detection
            if candidates:
                candidates = engine._detect_and_merge_dependencies(problem_text, candidates)

            return {
                'idx': problem_info['idx'],
                'actual_idx': actual_idx,
                'candidates': candidates,
                'pre_detected': False,
                'num_candidates': len(candidates)
            }

    def _format_detected_topics(self, detected_topics: Dict, domain: str) -> List[Dict]:
        """Format detected topics for storage."""
        formatted = []
        for topic_type, instances in detected_topics.items():
            for instance in instances:
                if len(instance) >= 2:
                    formatted.append({
                        'text': instance[0],
                        'position': instance[1],
                        'type': topic_type,
                        'domain': domain
                    })
        return formatted

    def _format_nl_entities(self, detected_entities: Dict) -> List[Dict]:
        """Format NL entities for storage."""
        formatted = []
        for entity_type, instances in detected_entities.items():
            for instance in instances:
                if len(instance) >= 2:
                    formatted.append({
                        'text': instance[0],
                        'original': instance[1] if len(instance) > 1 else instance[0],
                        'type': entity_type
                    })
        return formatted

    def _convert_pre_candidates_to_engine_format(self, pre_candidates: List[Dict]) -> Dict:
        """Convert pre-detected candidates back to engine format."""
        candidates = {
            'temporal': [],
            'math': [],
            'nl_entities': []
        }

        for candidate in pre_candidates:
            domain = candidate.get('domain', '')
            if domain == 'temporal':
                candidates['temporal'].append({
                    'text': candidate['text'],
                    'position': candidate['pos'],
                    'type': candidate['topic']
                })
            elif domain == 'math':
                candidates['math'].append({
                    'text': candidate['text'],
                    'position': candidate['pos'],
                    'type': candidate['topic']
                })
            elif domain == 'nl':
                candidates['nl_entities'].append({
                    'text': candidate['text'],
                    'original': candidate['pos'],
                    'type': candidate['topic']
                })

        return candidates









    def _process_variation_batch(self, batch_problems: List[Dict], start_idx: int):
        """Process a single batch of problems using cross-problem batching."""
        # SEMANTIC VERSION: Always use cross-problem batching (has semantic cluster logic)
        # Even for single problems - avoids old combination-based engine method
        use_cluster_variations = self.config.get('use_cluster_variations', True)
        if use_cluster_variations:
            self.logger.debug(f"  🔬 Semantic clustering: {len(batch_problems)} problem(s) with cluster-based variations")
        else:
            self.logger.debug(f"  📝 Processing: {len(batch_problems)} problem(s) (cluster variations disabled)")
        self._process_variation_batch_cross_problem(batch_problems, start_idx)

    def _process_variation_batch_cross_problem(self, batch_problems: List[Dict], start_idx: int):
        """Process problems using cross-problem batching for maximum API efficiency."""

        # Check if we need candidate detection at all
        use_cluster_variations = self.config.get('use_cluster_variations', True)

        if use_cluster_variations:
            self.logger.debug(f"    🔄 Phase 1: Candidate detection across {len(batch_problems)} problems...")
        else:
            self.logger.debug(f"    ⏭️  Phase 1: Skipping candidate detection (use_cluster_variations=False)...")

        # Phase 1: Extract all problems and detect candidates for all problems (only if needed)
        problem_data = []
        all_problem_candidates = []  # (problem_idx, candidate) pairs

        # Initialize variation engine once for the whole batch
        # Always create model client for variations stage (needed for combination-based variations)
        model_client = self._get_model_client_for_stage('variations')

        # TEST API CONNECTION (only for RITS, once per batch)
        if self._client_type == 'rits' and hasattr(model_client, 'test_api_connection'):
            if not hasattr(self, '_api_tested') or not self._api_tested:
                self.logger.debug(f"\n    🔍 Testing API connection before starting variation generation...")
                api_ok = model_client.test_api_connection()
                if not api_ok:
                    raise RuntimeError(f"❌ RITS API connection test failed! Cannot proceed with variation generation.")
                self._api_tested = True
                self.logger.debug(f"    ✅ API test passed - proceeding with variation generation\n")

        # Use variation model for validation (not judge model to avoid dual client initialization)
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
        engine = UnifiedVariationEngine(model_client=model_client, judge_model_client=None)

        # First pass: Extract problem info (fast, no detection yet)
        for i, problem in enumerate(batch_problems):
            actual_idx = start_idx + i

            # Extract problem data
            if isinstance(problem, dict):
                problem_text = problem.get('problem', problem.get('question', str(problem)))
                problem_id = problem.get('id', f'problem_{actual_idx}')
                ground_truth = problem.get('answer', problem.get('ground_truth', ''))
            else:
                problem_text = str(problem)
                problem_id = f'problem_{actual_idx}'
                ground_truth = ''

            problem_info = {
                'idx': i,
                'actual_idx': actual_idx,
                'problem_id': problem_id,
                'problem_text': problem_text,
                'ground_truth': ground_truth
            }
            problem_data.append(problem_info)

        # Second pass: Batch candidate detection (only if cluster variations are enabled)
        use_batch_detection = not (hasattr(self, '_candidates_pre_detected') and self._candidates_pre_detected)

        if use_cluster_variations and use_batch_detection:
            # ✅ BATCH DETECTION: Process all problems together (10-50x faster than multiprocessing)
            self.logger.debug(f"      🚀 Batch detecting candidates for {len(batch_problems)} problems...")

            # Extract problem texts
            problem_texts = [info['problem_text'] for info in problem_data]

            # Batch detect all candidates (engine already created above)
            all_candidates_batch = engine._detect_all_candidates_batch(problem_texts)

            # Apply dependency detection to each problem's candidates
            for i, candidates in enumerate(all_candidates_batch):
                if candidates:
                    all_candidates_batch[i] = engine._detect_and_merge_dependencies(problem_texts[i], candidates)

            # ═══════════════════════════════════════════════════════════════════
            # SEMANTIC CLUSTERING: Run clustering in Stage 1 as well (not just Stage 0)
            # ═══════════════════════════════════════════════════════════════════
            from benchdrift.pipeline.semantic_composite_detector import SemanticCompositeDetector, Candidate

            semantic_detector = SemanticCompositeDetector(
                embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2'),
                semantic_threshold=self.config.get('semantic_threshold', 0.35),
                verbose=self.verbose
            )

            self.logger.debug(f"      🔬 Running semantic clustering for {len(batch_problems)} problems...")
            all_clusters_batch = []

            for i, (problem_text, candidates) in enumerate(zip(problem_texts, all_candidates_batch)):
                if not candidates:
                    all_clusters_batch.append([])
                    continue

                # Convert to Candidate objects
                candidate_objects = []
                for cand in candidates:
                    if 'span' in cand:
                        span = cand['span']
                    elif 'pos' in cand:
                        start = cand['pos']
                        end = start + len(cand['text'])
                        span = (start, end)
                    else:
                        continue

                    candidate_objects.append(Candidate(
                        text=cand['text'],
                        span=span,
                        type=cand.get('topic', cand.get('domain', 'unknown'))
                    ))

                # Run clustering
                clusters = semantic_detector.detect_clusters(problem_text, candidate_objects)
                all_clusters_batch.append(clusters)

            # Store results with clusters
            for idx, (candidates, clusters) in enumerate(zip(all_candidates_batch, all_clusters_batch)):
                actual_idx = start_idx + idx
                num_candidates = len(candidates)
                num_clusters = len(clusters)
                self.logger.debug(f"      Problem {actual_idx + 1}: {num_candidates} candidates → {num_clusters} clusters")

                # Store clusters in self.data for later use in variation generation
                # Find or create entry for this problem
                problem_id = problem_data[idx]['problem_id']
                found = False
                for entry in self.data:
                    if entry.get('id') == problem_id:
                        # Serialize clusters
                        clusters_data = []
                        for cluster in clusters:
                            clusters_data.append({
                                'cluster_id': cluster.cluster_id,
                                'members': [
                                    {'text': m.text, 'span': list(m.span), 'type': m.type}
                                    for m in cluster.members
                                ]
                            })
                        entry['semantic_clusters'] = clusters_data
                        found = True
                        break

                if not found:
                    # Create new entry with clusters
                    clusters_data = []
                    for cluster in clusters:
                        clusters_data.append({
                            'cluster_id': cluster.cluster_id,
                            'members': [
                                {'text': m.text, 'span': list(m.span), 'type': m.type}
                                for m in cluster.members
                            ]
                        })
                    self.data.append({
                        'id': problem_id,
                        'problem': problem_texts[idx],
                        'semantic_clusters': clusters_data,
                        'is_baseline': True,  # This is the original problem
                        'is_variant': False  # Not a variation
                    })

                # Store (problem_idx, candidate) pairs for batching
                for candidate in candidates:
                    all_problem_candidates.append((idx, candidate))
        else:
            # Sequential path for pre-detected candidates (fast, no detection work)
            for i, problem in enumerate(batch_problems):
                actual_idx = start_idx + i
                pre_candidates = problem.get('candidates', [])
                candidates = self._convert_pre_candidates_to_engine_format(pre_candidates)
                self.logger.debug(f"      Problem {actual_idx + 1}: Using {len(pre_candidates)} pre-detected candidates")

                for candidate in candidates:
                    all_problem_candidates.append((i, candidate))

        # Only print candidate stats if cluster variations are enabled
        if use_cluster_variations:
            self.logger.debug(f"    📊 Total candidates across all problems: {len(all_problem_candidates)}")

        # Phase 2: Generate variations using original logic with cross-problem batching
        if use_cluster_variations:
            self.logger.debug(f"    🔄 Phase 2: Cross-problem batched variations (preserving original logic)...")
        else:
            self.logger.debug(f"    🔄 Phase 2: Generating variations (cluster variations disabled)...")
        all_variations_map = self._generate_variations_with_original_logic_batched(
            problem_data, all_problem_candidates, engine
        )

        # Phase 3: Organize results back by problem and create entries
        self.logger.debug(f"    🔄 Phase 3: Creating entries for all problems...")
        for problem_info in problem_data:
            problem_idx = problem_info['idx']

            # Get all variations for this specific problem from mega batch
            problem_variations = all_variations_map.get(problem_idx, [])

            self.logger.debug(f"      Problem {problem_info['actual_idx'] + 1}: {len(problem_variations)} total variations")

            # Create entries for this problem
            self._create_problem_entries(
                problem_info['problem_id'],
                problem_info['problem_text'],
                problem_info['ground_truth'],
                problem_variations
            )

    def _generate_variations_with_original_logic_batched(self, problem_data: List[Dict],
                                                       all_problem_candidates: List[tuple],
                                                       engine) -> Dict:
        """Generate variations using EXACT original logic + cross-problem batching where possible."""

        all_variations_map = {}
        use_cagrad = self.config.get('use_cagrad', False)
        # Configurable: user can choose to generate generic variations (default: True)
        use_generic = self.config.get('use_generic', True)

        # PART 1: Generate generic transformations (cross-problem batched for max efficiency)
        # User can enable/disable independently from CAGrad
        if use_generic:
            self.logger.debug(f"      🔄 Part 1: Generic transformations (cross-problem batched)...")
            generic_variations_map = self._batch_generic_transformations_cross_problem(problem_data, engine)
        else:
            self.logger.debug(f"      ⏭️  Part 1: Skipping generic transformations (use_generic=False)...")
            generic_variations_map = {}

        # PART 2: Generate cluster-based variations (using semantic clusters from Stage 0)
        use_cluster_variations = self.config.get('use_cluster_variations', True)
        if use_cluster_variations:
            self.logger.debug(f"      🔄 Part 2: Cluster-based variations (using semantic clusters)...")
            cluster_variations_map = self._batch_combination_variations_with_original_logic(
                problem_data, all_problem_candidates, engine
            )
        else:
            self.logger.debug(f"      ⏭️  Part 2: Skipping cluster-based variations (use_cluster_variations=False)...")
            cluster_variations_map = {}

        # PART 3: Generate long context variations (for prompts >500 chars)
        use_long_context = self.config.get('use_long_context', False)
        if use_long_context:
            self.logger.debug(f"      🔄 Part 3: Long context variations (for prompts >500 chars)...")
            long_context_variations_map = self._batch_long_context_variations(problem_data, engine)
        else:
            self.logger.debug(f"      ⏭️  Part 3: Skipping long context variations (use_long_context=False)...")
            long_context_variations_map = {}

        # Combine all results by problem
        for problem_idx, problem_info in enumerate(problem_data):
            all_variations_map[problem_idx] = []

            # Add generic variations
            if problem_idx in generic_variations_map:
                all_variations_map[problem_idx].extend(generic_variations_map[problem_idx])

            # Add cluster variations
            if problem_idx in cluster_variations_map:
                all_variations_map[problem_idx].extend(cluster_variations_map[problem_idx])

            # Add long context variations
            if problem_idx in long_context_variations_map:
                all_variations_map[problem_idx].extend(long_context_variations_map[problem_idx])

        return all_variations_map

    def _batch_generic_transformations_cross_problem(self, problem_data: List[Dict], engine) -> Dict:
        """Generate generic transformations with cross-problem batching."""
        self.logger.debug(f"        🚀 Batching generic transformations across {len(problem_data)} problems...")

        # ✅ Get ALL transformation types from engine (includes generic + personas!)
        from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
        all_transformation_types = UnifiedVariationEngine.get_all_transformation_types()

        # Filter based on flags
        transformation_types = {}
        use_generic = self.config.get('use_generic', True)
        use_persona = self.config.get('use_persona', False)

        for trans_type, config in all_transformation_types.items():
            is_persona = trans_type.endswith('_persona')

            # Include if:
            # - It's a persona and use_persona is True, OR
            # - It's NOT a persona and use_generic is True
            if (is_persona and use_persona) or (not is_persona and use_generic):
                transformation_types[trans_type] = config

        # Count persona vs generic types
        persona_count = sum(1 for t in transformation_types.keys() if t.endswith('_persona'))
        generic_count = len(transformation_types) - persona_count
        self.logger.debug(f"        📋 Using {generic_count} generic + {persona_count} persona types ({len(transformation_types)} total)...")
        if not use_generic:
            self.logger.debug(f"        ⚠️  Generic transformations DISABLED")
        if not use_persona:
            self.logger.debug(f"        ⚠️  Persona transformations DISABLED")

        # Collect all generic transformation requests
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []

        for problem_idx, problem_info in enumerate(problem_data):
            problem_text = problem_info['problem_text']

            for trans_type, config in transformation_types.items():
                system_prompt = f"""You are an expert at creating question variations that test specific cognitive capabilities.

TASK: Create a {trans_type} variation of the given problem.

TRANSFORMATION GOAL: {config['prompt']}

CRITICAL REQUIREMENTS:
1. PRESERVE the exact numerical answer - the final result must be IDENTICAL
2. MAINTAIN all mathematical relationships and calculation methods
3. Use PLAIN TEXT only - no markdown formatting
4. Ensure the variation tests {config['capability']} capability
5. Make meaningful changes while keeping the same numerical answer

OUTPUT FORMAT: Return ONLY the transformed question wrapped in <question> tags.
Example: <question>Your transformed question here</question>"""

                # Include examples if available (for better quality, especially for personas)
                examples_text = ""
                if 'examples' in config and config['examples']:
                    examples_text = f"\n\nExamples of this transformation type:\n"
                    examples_text += "\n".join(f"• {ex}" for ex in config['examples'])

                user_prompt = f"""Original Problem: {problem_text}

Create a {trans_type} variation that {config['prompt']}{examples_text}

Generate ONE high-quality {trans_type} variation."""

                all_system_prompts.append(system_prompt)
                all_user_prompts.append(user_prompt)
                request_mapping.append((problem_idx, trans_type, config))

        # Execute batch call (chunked by batch_size)
        generic_variations_map = {}
        try:
            if hasattr(engine.model_client, 'get_model_response'):
                # Chunk prompts by batch_size to respect user configuration
                responses = []
                total_prompts = len(all_system_prompts)
                for chunk_start in range(0, total_prompts, self.batch_size):
                    chunk_end = min(chunk_start + self.batch_size, total_prompts)
                    chunk_sys = all_system_prompts[chunk_start:chunk_end]
                    chunk_user = all_user_prompts[chunk_start:chunk_end]
                    self.logger.debug(f"          Processing generic variations batch {chunk_start//self.batch_size + 1}/{(total_prompts + self.batch_size - 1)//self.batch_size} ({len(chunk_sys)} prompts)...")
                    chunk_responses = engine.model_client.get_model_response(chunk_sys, chunk_user)
                    responses.extend(chunk_responses)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 5 == 0:
                        self.logger.debug(f"          Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses
            successful_transformations = 0
            for i, (response, (problem_idx, trans_type, config)) in enumerate(zip(responses, request_mapping)):
                import re
                question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
                if question_match:
                    from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question
                    cleaned = clean_model_response(question_match.group(1).strip())

                    if cleaned and is_valid_question(cleaned):
                        variation = {
                            'original_problem': problem_data[problem_idx]['problem_text'],
                            'modified_problem': cleaned,
                            'transformation_type': f'cross_batch_generic_{trans_type}',
                            'generation_method': 'cross_problem_generic_batched',
                            'detection_method': 'pattern_based',
                            'debugging_capability': config['capability'],
                            'confidence': 'cross_batch_generated',
                            'domains_involved': ['generic'],
                            'original_component': trans_type,
                            'new_component': f'{trans_type}_variation',
                            'combination_size': 0,
                            'cross_domain': False
                        }

                        if problem_idx not in generic_variations_map:
                            generic_variations_map[problem_idx] = []
                        generic_variations_map[problem_idx].append(variation)
                        successful_transformations += 1

            self.logger.debug(f"        ✅ Generic batch: {successful_transformations}/{len(request_mapping)} successful")

            # Note: Validation moved to separate stage 1a (validation stage)
            total_generated = sum(len(v) for v in generic_variations_map.values())
            self.logger.debug(f"        ✅ Generic variations generated: {total_generated} variations (validation will be done in stage 1a)")

        except Exception as e:
            self.logger.debug(f"        ❌ Generic batch failed: {e}")
            import traceback
            traceback.print_exc()

        return generic_variations_map

    def _batch_long_context_variations(self, problem_data: List[Dict], engine) -> Dict:
        """Generate long context variations for prompts >500 chars."""
        self.logger.debug(f"        🚀 Batching long context variations across {len(problem_data)} problems...")

        # Import long context variations
        from benchdrift.pipeline.long_context_variations import (
            get_long_context_transformation_types,
            is_long_context,
            LongContextStructure
        )

        # Filter problems that qualify as long context
        long_context_problems = []
        problem_idx_map = {}
        for problem_idx, problem_info in enumerate(problem_data):
            problem_text = problem_info['problem_text']
            if is_long_context(problem_text):
                long_context_problems.append((problem_idx, problem_text, problem_info.get('expected_answer', '')))
                problem_idx_map[len(long_context_problems) - 1] = problem_idx

        if not long_context_problems:
            self.logger.debug(f"        ℹ️  No long context problems found (need >500 chars)")
            return {}

        self.logger.debug(f"        📋 Found {len(long_context_problems)} long context problems (out of {len(problem_data)} total)")

        # Get transformation types
        transformation_types = get_long_context_transformation_types()

        # Collect all variation requests
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []

        # Separate handling for deterministic vs LLM variations
        deterministic_variations = []

        for lc_idx, (problem_idx, problem_text, expected_answer) in enumerate(long_context_problems):
            # Parse structure
            structure = LongContextStructure(problem_text)

            for trans_type, config in transformation_types.items():
                trans_config = config
                var_type = trans_config.get('type', 'llm')

                if var_type == 'deterministic':
                    # Deterministic variations: Apply function directly
                    if 'variants' in trans_config:
                        for variant in trans_config['variants']:
                            try:
                                modified_text = variant['function'](problem_text)
                                deterministic_variations.append({
                                    'problem_idx': problem_idx,
                                    'trans_type': f"{trans_type}.{variant['name']}",
                                    'capability': variant.get('capability', 'format_robustness'),
                                    'modified_problem': modified_text
                                })
                            except Exception as e:
                                self.logger.debug(f"          ⚠️  Deterministic variation {trans_type}.{variant['name']} failed: {e}")

                elif var_type == 'llm':
                    # LLM-based variations
                    if 'variants' in trans_config:
                        # Multiple variants (e.g., redundancy.remove, redundancy.add)
                        for variant in trans_config['variants']:
                            system_prompt = f"""You are an expert at creating long-context question variations that preserve the answer.

TASK: Apply {trans_type}.{variant['name']} transformation.

CRITICAL REQUIREMENTS:
1. PRESERVE the exact answer - it must remain: {expected_answer}
2. All numerical values, facts, and logic NEEDED TO DERIVE {expected_answer} must remain unchanged
3. Verify that your modified context still allows deriving {expected_answer}
4. Use PLAIN TEXT only - no markdown formatting
5. Output wrapped in <context> and <query> tags

OUTPUT FORMAT:
<context>Your transformed context here</context>
<query>Your query here</query>"""

                            user_prompt = variant['prompt'].format(
                                context=structure.context,
                                query=structure.query,
                                answer=expected_answer
                            )

                            all_system_prompts.append(system_prompt)
                            all_user_prompts.append(user_prompt)
                            request_mapping.append((problem_idx, f"{trans_type}.{variant['name']}", variant.get('capability', 'long_context')))

                    else:
                        # Single prompt variation
                        if 'prompt' in trans_config:
                            system_prompt = f"""You are an expert at creating long-context question variations that preserve the answer.

TASK: Apply {trans_type} transformation.

CRITICAL REQUIREMENTS:
1. PRESERVE the exact answer - it must remain: {expected_answer}
2. All numerical values, facts, and logic NEEDED TO DERIVE {expected_answer} must remain unchanged
3. Verify that your modified context still allows deriving {expected_answer}
4. Use PLAIN TEXT only - no markdown formatting
5. Output wrapped in <context> and <query> tags

OUTPUT FORMAT:
<context>Your transformed context here</context>
<query>Your query here</query>"""

                            user_prompt = trans_config['prompt'].format(
                                context=structure.context,
                                query=structure.query,
                                answer=expected_answer
                            )

                            all_system_prompts.append(system_prompt)
                            all_user_prompts.append(user_prompt)
                            request_mapping.append((problem_idx, trans_type, trans_config.get('capability', 'long_context')))

        # Process deterministic variations immediately
        long_context_variations_map = {}
        if deterministic_variations:
            self.logger.debug(f"        ✅ Applied {len(deterministic_variations)} deterministic format variations")
            for det_var in deterministic_variations:
                problem_idx = det_var['problem_idx']
                if problem_idx not in long_context_variations_map:
                    long_context_variations_map[problem_idx] = []

                variation = {
                    'original_problem': problem_data[problem_idx]['problem_text'],
                    'modified_problem': det_var['modified_problem'],
                    'transformation_type': det_var["trans_type"],
                    'generation_method': 'deterministic',
                    'detection_method': 'format_based',
                    'debugging_capability': det_var['capability'],
                    'confidence': 'deterministic',
                    'domains_involved': ['format'],
                }
                long_context_variations_map[problem_idx].append(variation)

        # Execute batch call for LLM-based variations (chunked by batch_size)
        if len(all_system_prompts) == 0:
            self.logger.debug(f"        ℹ️  No long context variations to generate")
            return {}

        long_context_variations_map = {}

        try:
            if hasattr(engine.model_client, 'get_model_response'):
                # Chunk prompts by batch_size
                responses = []
                total_prompts = len(all_system_prompts)
                for chunk_start in range(0, total_prompts, self.batch_size):
                    chunk_end = min(chunk_start + self.batch_size, total_prompts)
                    chunk_sys = all_system_prompts[chunk_start:chunk_end]
                    chunk_user = all_user_prompts[chunk_start:chunk_end]
                    self.logger.debug(f"          Processing long context batch {chunk_start//self.batch_size + 1}/{(total_prompts + self.batch_size - 1)//self.batch_size} ({len(chunk_sys)} prompts)...")
                    chunk_responses = engine.model_client.get_model_response(chunk_sys, chunk_user)
                    responses.extend(chunk_responses)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 5 == 0:
                        self.logger.debug(f"          Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses
            successful_transformations = 0
            for i, (response, (problem_idx, trans_type, capability)) in enumerate(zip(responses, request_mapping)):
                import re
                # Try to extract context and query
                context_match = re.search(r'<context>(.*?)</context>', response, re.DOTALL)
                query_match = re.search(r'<query>(.*?)</query>', response, re.DOTALL)

                if context_match and query_match:
                    from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question
                    context_text = clean_model_response(context_match.group(1).strip())
                    query_text = clean_model_response(query_match.group(1).strip())

                    # Combine context + query
                    modified_problem = f"{context_text}\n\n{query_text}"

                    if modified_problem and is_valid_question(modified_problem):
                        variation = {
                            'original_problem': problem_data[problem_idx]['problem_text'],
                            'modified_problem': modified_problem,
                            'transformation_type': trans_type,
                            'generation_method': 'long_context_batched',
                            'detection_method': 'structure_based',
                            'debugging_capability': capability,
                            'confidence': 'long_context_generated',
                            'domains_involved': ['long_context'],
                            'original_component': trans_type,
                            'new_component': f'{trans_type}_variation',
                            'combination_size': 0,
                            'cross_domain': False
                        }

                        if problem_idx not in long_context_variations_map:
                            long_context_variations_map[problem_idx] = []
                        long_context_variations_map[problem_idx].append(variation)
                        successful_transformations += 1

            self.logger.debug(f"        ✅ Long context batch: {successful_transformations}/{len(request_mapping)} successful")

            total_generated = sum(len(v) for v in long_context_variations_map.values())
            self.logger.debug(f"        ✅ Long context variations total: {total_generated} LLM-based variations")

        except Exception as e:
            self.logger.debug(f"        ❌ Long context batch failed: {e}")
            import traceback
            traceback.print_exc()

        return long_context_variations_map

    def _generate_generic_per_problem(self, problem_data: List[Dict], engine) -> Dict:
        """Generate generic transformations using the engine (proper architecture)."""
        self.logger.debug(f"        🚀 Using engine for generic transformations...")

        generic_variations_map = {}

        for problem_idx, problem_info in enumerate(problem_data):
            problem_text = problem_info['problem_text']

            try:
                # Let the engine handle generic transformations (proper architecture)
                generic_variations = engine._generate_generic_transformations(problem_text)

                if generic_variations:
                    generic_variations_map[problem_idx] = generic_variations
                    self.logger.debug(f"          Problem {problem_idx + 1}: {len(generic_variations)} generic variations")

            except Exception as e:
                self.logger.debug(f"          ⚠️ Generic variations failed for problem {problem_idx + 1}: {e}")
                continue

        total_generic = sum(len(variations) for variations in generic_variations_map.values())
        self.logger.debug(f"        ✅ Engine generated {total_generic} total generic variations")

        return generic_variations_map

    def _batch_combination_variations_with_original_logic(self, problem_data: List[Dict],
                                                        all_problem_candidates: List[tuple],
                                                        engine) -> Dict:
        """
        Generate combination variations using DIRECT VARIATION approach with cross-problem batching.

        ✅ NEW APPROACH WITH CAGrad (when use_cagrad=True):
        - Step 1: Run CAGrad brittleness analysis (generates counterfactuals automatically)
        - Step 2: Harvest pre-generated counterfactuals as variations

        ✅ DIRECT VARIATION APPROACH (when use_cagrad=False, default):
        - Step 1: Select diverse combinations (asks model to choose best combinations)
        - Step 2: Generate complete variations directly (no bucket/substitution)

        """
        # SEMANTIC VERSION: Use clusters from Stage 0 (no old combination logic)
        self.logger.debug(f"        🔬 Using CLUSTER-BASED approach (semantic clusters from Stage 0)...")

        # Load clusters from Stage 0 for each problem
        all_selected_combinations = {}
        for problem_idx, problem_info in enumerate(problem_data):
            problem_id = problem_info['problem_id']

            # Find clusters for this problem in self.data
            clusters = []
            for entry in self.data:
                if entry.get('id') == problem_id and entry.get('semantic_clusters'):
                    clusters = entry['semantic_clusters']
                    break

            if not clusters:
                continue

            # Convert clusters to "combination" format (same structure as original)
            problem_combinations = []
            for cluster in clusters:
                cluster_id = cluster['cluster_id']
                members = cluster['members']

                if not members:
                    continue

                # Create combination from cluster members (format expected by engine)
                # Convert cluster members to candidate format: {'text', 'pos', 'domain', 'topic'}
                candidates = []
                for m in members:
                    candidates.append({
                        'text': m['text'],
                        'pos': m['span'][0] if isinstance(m['span'], list) else m['span'],
                        'span': m['span'],
                        'domain': m.get('type', 'unknown'),
                        'topic': m.get('type', 'unknown')
                    })

                combination = {
                    'candidates': candidates,
                    'elements': [m['text'] for m in members],
                    'domains': list(set(m.get('type', 'unknown') for m in members)),
                    'size': len(members),
                    'cluster_id': cluster_id
                }
                problem_combinations.append(combination)

            if problem_combinations:
                all_selected_combinations[problem_idx] = problem_combinations
                self.logger.debug(f"        Problem {problem_idx + 1}: {len(problem_combinations)} clusters → combinations")

        if not all_selected_combinations:
            self.logger.debug(f"        ℹ️ No clusters found from Stage 0")
            return {}

        self.logger.debug(f"        📊 {len(all_selected_combinations)} problems with semantic clusters")

        # STEP 2: Cross-problem batch DIRECT variation generation (using clusters as combinations)
        self.logger.debug(f"        🚀 Step 2: Cross-problem batching DIRECT variation generation...")
        combination_variations_map = self._batch_direct_variations_cross_problem(
            problem_data, all_selected_combinations, engine
        )

        return combination_variations_map

    def _batch_direct_variations_cross_problem(self, problem_data: List[Dict],
                                             all_selected_combinations: Dict, engine) -> Dict:
        """
        Call engine's cross-problem batching method for DIRECT variation generation.

        This method orchestrates the call to the engine and organizes results.
        Core logic lives in: unified_variation_engine_batched._generate_direct_variations_cross_problem_batched()
        """

        # Prepare data for engine's cross-problem batching method
        problems_with_combinations = []
        for problem_idx in all_selected_combinations.keys():
            if all_selected_combinations[problem_idx]:
                problem_text = problem_data[problem_idx]['problem_text']
                ground_truth = problem_data[problem_idx]['ground_truth']
                selected_combinations = all_selected_combinations[problem_idx]
                problems_with_combinations.append((problem_idx, problem_text, ground_truth, selected_combinations))

        if not problems_with_combinations:
            self.logger.debug(f"          ℹ️  No problems with selected combinations for direct variation generation")
            return {}

        # CALL ENGINE: This is where the core logic lives
        self.logger.debug(f"          📞 Calling engine's cross-problem direct variation method...")
        combination_variations_map = engine._generate_direct_variations_cross_problem_batched(
            problems_with_combinations,
            rectify_invalid=self.rectify_invalid
        )

        self.logger.debug(f"          ✅ Received variations for {len(combination_variations_map)} problems from engine")
        return combination_variations_map

    # ═══════════════════════════════════════════════════════════════════════════════
    # CAGrad INTEGRATION - BRITTLENESS-BASED VARIATION GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    def _run_cagrad_scoring(self, problem_data: List[Dict], engine) -> Dict:
        """
        Run CAGrad candidate detection and pruning on batch of problems.

        Uses UnifiedVariationEngine for sophisticated composite-aware detection
        and gradient-based pruning.

        Returns:
            Dict[int, List[CandidateFragment]]: Map from problem_idx to pruned candidates
        """
        from benchdrift.pipeline.cagrad_fragment_scorer import CAGradFragmentScorer

        self.logger.debug(f"        🎯 Running CAGrad candidate detection and pruning...")

        # Extract problem texts and answers
        problem_texts = [info['problem_text'] for info in problem_data]
        expected_answers = [info.get('answer', '') for info in problem_data]

        # Get config for CAGrad (parameterized, not hardcoded!)
        use_gradient_pruning = self.config.get('cagrad_use_gradient_pruning', True)
        pruning_threshold = self.config.get('cagrad_pruning_threshold', 0.3)
        min_candidates = self.config.get('cagrad_min_fragments', 3)  # Configurable min
        max_candidates = self.config.get('cagrad_max_fragments', None)  # Configurable max

        # Create scorer
        scorer = CAGradFragmentScorer(
            model_client=engine.model_client,
            variation_engine=engine,  # Pass the full engine for composite detection!
            use_gradient_pruning=use_gradient_pruning
        )

        # Detect and prune candidates for all problems in batch
        cagrad_results = scorer.detect_and_prune_batch(
            problems=problem_texts,
            expected_answers=expected_answers,
            pruning_threshold=pruning_threshold,
            min_candidates=min_candidates,
            max_candidates=max_candidates
        )

        # Print summary
        total_candidates = sum(len(cands) for cands in cagrad_results.values())
        avg_gradient = sum(
            sum(c.gradient_score for c in cands) / len(cands) if cands else 0
            for cands in cagrad_results.values()
        ) / len(cagrad_results) if cagrad_results else 0

        self.logger.debug(f"        ✅ CAGrad detected and pruned {total_candidates} candidates across {len(problem_data)} problems")
        self.logger.debug(f"           Average gradient score: {avg_gradient:.4f}")
        self.logger.debug(f"           Pruning threshold: {pruning_threshold} (keep top {int(pruning_threshold*100)}%)")
        self.logger.debug(f"           Min candidates: {min_candidates}, Max candidates: {max_candidates if max_candidates else 'unlimited'}")

        return cagrad_results

    def _harvest_variations_from_cagrad(self, problem_data: List[Dict],
                                       cagrad_results: Dict[int, List]) -> Dict[int, List[Dict]]:
        """
        Convert CAGrad pruned candidates into combinations for variation generation.

        CAGrad provides pruned candidates with gradient scores. This method converts
        them into the pipeline's standard combination format so the pipeline can
        generate variations using its model-based approach.

        Args:
            problem_data: List of problem info dicts
            cagrad_results: Dict mapping problem_idx to List[CandidateFragment]

        Returns:
            Dict[int, List[Dict]]: Map from problem_idx to list of candidate combinations
        """
        self.logger.debug(f"        🌾 Converting CAGrad candidates to variation combinations...")

        combinations_map = {}
        total_candidates = 0

        for problem_idx, candidates in cagrad_results.items():
            if problem_idx >= len(problem_data):
                continue

            # Convert CandidateFragments to combination format
            # Each candidate becomes a single-element combination for variation generation
            combinations = []
            for candidate in candidates:
                # Validate candidate has required fields
                if not hasattr(candidate, 'text') or not candidate.text:
                    continue
                if not hasattr(candidate, 'domain') or not candidate.domain:
                    continue

                # Convert CandidateFragment to candidate dict
                candidate_dict = {
                    'text': candidate.text,
                    'domain': candidate.domain,
                    'topic': getattr(candidate, 'topic', ''),
                    'span': getattr(candidate, 'span', None),
                    'type': getattr(candidate, 'type', ''),
                    'is_composite': getattr(candidate, 'is_composite', False),
                    'gradient_score': getattr(candidate, 'gradient_score', 0.0),
                    'priority': getattr(candidate, 'priority', 0),
                }

                # Create combination dict in CORRECT format (same as non-CAGrad path)
                # All required fields explicitly set
                combination = {
                    'candidates': [candidate_dict],  # Must have 'candidates' field as list!
                    'size': 1,  # Required by engine - always 1 for single-element
                    'combination_size': 1,  # Single candidate
                    'cross_domain': False,  # Single candidate can't be cross-domain
                    'reason': f'High gradient importance (score: {candidate.gradient_score:.3f}) - varying this {candidate.domain} element significantly affects model prediction',  # Required by engine
                    # Metadata for tracking
                    'generation_method': 'cagrad_gradient_guided',
                    'debugging_capability': f'gradient_importance_{candidate.domain}',
                }
                combinations.append(combination)
                total_candidates += 1

            if combinations:
                combinations_map[problem_idx] = combinations
                self.logger.debug(f"          Problem {problem_idx + 1}: {len(combinations)} candidates (gradient-pruned)")

        self.logger.debug(f"        ✅ Converted {total_candidates} CAGrad candidates to variation combinations")
        return combinations_map

    def _convert_cagrad_to_candidate_format(self, cagrad_results: Dict[int, List]) -> Dict[int, List[Dict]]:
        """
        Convert CAGrad pruned candidates back to standard candidate format.

        This is needed so we can use model-based combination selection on the
        gradient-pruned candidates.

        IMPORTANT: Deduplicates candidates by text, keeping the one with highest gradient score.
        This prevents combinations like ["9:00 AM", "9:00 AM", "9:00"] where the same text
        appears multiple times due to different spans in the original problem.

        Args:
            cagrad_results: Dict mapping problem_idx to List[CandidateFragment]

        Returns:
            Dict[int, List[Dict]]: Map from problem_idx to list of UNIQUE candidate dicts
        """
        candidates_by_problem = {}

        for problem_idx, pruned_candidates in cagrad_results.items():
            # Use dict to track best candidate for each unique text
            best_candidates_by_text = {}

            for candidate in pruned_candidates:
                candidate_text = candidate.text
                gradient_score = candidate.gradient_score if hasattr(candidate, 'gradient_score') else 0.0

                # If we haven't seen this text, or this one has higher gradient, keep it
                if candidate_text not in best_candidates_by_text:
                    best_candidates_by_text[candidate_text] = candidate
                elif gradient_score > getattr(best_candidates_by_text[candidate_text], 'gradient_score', 0.0):
                    best_candidates_by_text[candidate_text] = candidate

            # Convert best candidates to dict format
            candidates = []
            for candidate in best_candidates_by_text.values():
                candidate_dict = {
                    'text': candidate.text,
                    'domain': candidate.domain,
                    'topic': candidate.topic,
                    'span': candidate.span,
                    'type': candidate.type,
                    'is_composite': candidate.is_composite,
                    'gradient_score': candidate.gradient_score,
                    'priority': candidate.priority,
                }
                candidates.append(candidate_dict)

            if candidates:
                original_count = len(pruned_candidates)
                deduplicated_count = len(candidates)
                duplicates_removed = original_count - deduplicated_count

                candidates_by_problem[problem_idx] = candidates

                if duplicates_removed > 0:
                    self.logger.debug(f"          Problem {problem_idx + 1}: {deduplicated_count} unique candidates ready for model selection [{duplicates_removed} duplicate texts removed]")
                else:
                    self.logger.debug(f"          Problem {problem_idx + 1}: {deduplicated_count} pruned candidates ready for model selection")

        return candidates_by_problem

    def _merge_combinations(self, single_element_combinations: Dict[int, List[Dict]],
                           multi_element_combinations: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        """
        Merge single-element and multi-element combinations for each problem.

        Args:
            single_element_combinations: Dict mapping problem_idx to single-element combinations
            multi_element_combinations: Dict mapping problem_idx to multi-element combinations

        Returns:
            Dict[int, List[Dict]]: Merged combinations by problem
        """
        merged = {}

        # Check if we should filter redundant singles (configurable via --filter-redundant-singles)
        filter_redundant_singles = self.config.get('filter_redundant_singles', False)

        # Get all problem indices
        all_problem_indices = set(single_element_combinations.keys()) | set(multi_element_combinations.keys())

        for problem_idx in all_problem_indices:
            merged[problem_idx] = []
            seen_combinations = set()

            # STEP 1: Collect all candidate texts that appear in multi-element combinations
            # (Only if filtering is enabled)
            candidates_in_multi = set()
            if filter_redundant_singles and problem_idx in multi_element_combinations:
                for combo in multi_element_combinations[problem_idx]:
                    if combo and 'candidates' in combo and combo['candidates']:
                        for candidate in combo['candidates']:
                            if candidate and 'text' in candidate and candidate['text']:
                                candidates_in_multi.add(candidate['text'])

            # STEP 2: Add multi-element combinations first (higher priority - more informative)
            duplicates_filtered = 0
            invalid_filtered = 0
            if problem_idx in multi_element_combinations:
                for combo in multi_element_combinations[problem_idx]:
                    # Validate combination structure
                    if not combo or 'candidates' not in combo or not combo['candidates']:
                        invalid_filtered += 1
                        continue
                    if not all(c and 'text' in c and c['text'] for c in combo['candidates']):
                        invalid_filtered += 1
                        continue

                    combo_key = frozenset(c['text'] for c in combo['candidates'])
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        merged[problem_idx].append(combo)
                    else:
                        duplicates_filtered += 1

            # STEP 3: Add single-element combinations, but ONLY if candidate NOT in multi-element
            # This avoids redundancy: if we test candidate_A with candidate_B, no need to test candidate_A alone
            singles_filtered = 0
            if problem_idx in single_element_combinations:
                for combo in single_element_combinations[problem_idx]:
                    # Validate combination structure
                    if not combo or 'candidates' not in combo or not combo['candidates']:
                        continue
                    if not all(c and 'text' in c and c['text'] for c in combo['candidates']):
                        continue

                    # Check if this single-element candidate appears in any multi-element combination
                    candidate = combo['candidates'][0]  # Single-element has exactly one candidate
                    candidate_text = candidate['text']

                    if candidate_text in candidates_in_multi:
                        singles_filtered += 1
                        continue  # Skip - this candidate is already tested in multi-element combinations

                    # Track this combination
                    combo_key = frozenset(c['text'] for c in combo['candidates'])
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        merged[problem_idx].append(combo)

            single_count = len(single_element_combinations.get(problem_idx, []))
            multi_count = len(multi_element_combinations.get(problem_idx, []))
            total_count = len(merged[problem_idx])

            # Build info message about filtering
            filter_info = []
            if singles_filtered > 0:
                filter_info.append(f"{singles_filtered} redundant singles")
            if duplicates_filtered > 0:
                filter_info.append(f"{duplicates_filtered} duplicates")
            if invalid_filtered > 0:
                filter_info.append(f"{invalid_filtered} invalid")

            if filter_info:
                filter_str = " [" + ", ".join(filter_info) + " filtered]"
                self.logger.debug(f"          Problem {problem_idx + 1}: {single_count} single + {multi_count} multi = {total_count} total combinations{filter_str}")
            else:
                self.logger.debug(f"          Problem {problem_idx + 1}: {single_count} single + {multi_count} multi = {total_count} total combinations")

        return merged

    # ═══════════════════════════════════════════════════════════════════════════════
    # STAGE 2 OVERRIDE - PRIORITIZED RESPONSE GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    def stage2_generate_responses(self, eval_model=None, force_regenerate=False):
        """
        Stage 2: Generate model responses with optional prioritized testing.

        If use_prioritized_testing is enabled:
        1. Score all variations with logprobs (fast)
        2. Only generate responses for top N% by drift likelihood
        3. Mark skipped variations for potential future testing

        This provides significant speedup by avoiding response generation
        for low-priority variations.
        """
        # No prioritized testing in semantic version - process all variations
        # Call parent's stage2 method
        super().stage2_generate_responses(eval_model, force_regenerate)

    # ═══════════════════════════════════════════════════════════════════════════════
    # PRIORITIZED TESTING - LOGPROB-BASED DRIFT PREDICTION FOR SPEED
    # ═══════════════════════════════════════════════════════════════════════════════

    def _score_variation_drift_likelihood(self, problem: str, baseline_answer: str,
                                          variation_problem: str, model_client) -> float:
        """
        Score how likely a variation is to cause drift using logprobs.

        CRITICAL: This checks if the model can still produce the CORRECT ANSWER
        when given the variation. Drift = model fails to produce correct answer.

        Approach (same as CAGrad):
        1. Generate answer for baseline, check if matches target
        2. Generate answer for variation, check if matches target
        3. Penalize probability if answer is wrong
        4. Drift likelihood = difference in probabilities

        Args:
            problem: Original baseline problem
            baseline_answer: Ground truth answer (CRITICAL - this is what we check against!)
            variation_problem: Variation to score
            model_client: Model client with logprobs support

        Returns:
            Drift likelihood score (0-1, higher = more likely to drift)
        """
        try:
            import numpy as np
            system_prompt = "You are a helpful assistant. Solve the problem and provide only the final answer."

            # Get baseline: P(correct_answer | original_problem)
            baseline_result = model_client.get_model_response_with_logprobs(
                [system_prompt],
                [problem],
                max_new_tokens=50,
                temperature=0.1
            )

            # Get variation: P(correct_answer | variation)
            variation_result = model_client.get_model_response_with_logprobs(
                [system_prompt],
                [variation_problem],
                max_new_tokens=50,
                temperature=0.1
            )

            if not baseline_result or not variation_result:
                return 0.5  # Unknown - medium priority

            # Extract generated text and logprobs
            baseline_text = baseline_result[0].get('text', '').strip()
            baseline_logprobs = baseline_result[0].get('logprobs', [])

            variation_text = variation_result[0].get('text', '').strip()
            variation_logprobs = variation_result[0].get('logprobs', [])

            if not baseline_logprobs or not variation_logprobs:
                return 0.5  # Unknown

            # CRITICAL: Check if generated answers match the target answer
            baseline_matches = self._fuzzy_match_answer(baseline_text, baseline_answer)
            variation_matches = self._fuzzy_match_answer(variation_text, baseline_answer)

            # Compute average log probabilities
            baseline_avg_logprob = float(np.mean(baseline_logprobs))
            variation_avg_logprob = float(np.mean(variation_logprobs))

            # Convert to probabilities
            baseline_prob = np.exp(baseline_avg_logprob)
            variation_prob = np.exp(variation_avg_logprob)

            # CRITICAL: Penalize if answer doesn't match target
            if not baseline_matches:
                baseline_prob *= 0.1  # Heavily penalize wrong baseline
            if not variation_matches:
                variation_prob *= 0.1  # Heavily penalize wrong variation

            # Drift likelihood based on probability drop
            # If baseline is correct but variation is wrong, this will be high
            # If both are wrong or both are right with similar confidence, this will be low
            if baseline_prob > 0:
                prob_ratio = variation_prob / baseline_prob
                # Convert ratio to drift score:
                # ratio near 1.0 = similar performance = low drift
                # ratio near 0.0 = variation much worse = high drift
                drift_score = 1.0 - prob_ratio
                drift_score = max(0.0, min(1.0, drift_score))  # Clamp to [0,1]
            else:
                # Baseline is already failing, hard to measure drift
                drift_score = 0.5

            return float(drift_score)

        except Exception as e:
            self.logger.debug(f"      ⚠️ Scoring failed: {e}")
            return 0.5  # Unknown - medium priority

    def _fuzzy_match_answer(self, generated: str, expected: str) -> bool:
        """
        Fuzzy match between generated and expected answer.
        Reuses CAGrad's matching logic.
        """
        # Normalize
        gen = generated.lower().strip()
        exp = expected.lower().strip()

        # Exact match
        if gen == exp:
            return True

        # Check if expected is in generated
        if exp in gen:
            return True

        # Extract numbers and compare
        import re
        gen_nums = set(re.findall(r'\d+\.?\d*', gen))
        exp_nums = set(re.findall(r'\d+\.?\d*', exp))

        if gen_nums and exp_nums and gen_nums == exp_nums:
            return True

        return False

    def _compute_heuristic_drift_score(self, variation: Dict) -> float:
        """
        Fast heuristic drift scoring (ZERO API calls).

        For non-CAGrad variations (generic, persona, etc.), use simple heuristics:
        - Combination size: larger = more likely to drift
        - Variation type: some types drift more than others
        - Edit distance: larger change = more likely to drift

        Returns:
            Drift score (0-1, higher = more likely to drift)
        """
        score = 0.5  # Default: medium priority

        # Heuristic 1: Combination size (multi-element changes drift more)
        size = variation.get('size', variation.get('combination_size', 1))
        if size >= 4:
            score += 0.3
        elif size >= 2:
            score += 0.2

        # Heuristic 2: Variation type
        transformation_type = variation.get('transformation_type', '')
        if transformation_type in ['cross_domain', 'multi_candidate']:
            score += 0.2  # Cross-domain tends to drift more
        elif transformation_type in ['persona', 'rephrasing']:
            score -= 0.1  # These tend to be stable

        # Heuristic 3: Edit distance (if available)
        original = variation.get('baseline_problem', '')
        modified = variation.get('modified_problem', '')
        if original and modified:
            # Simple character-level diff
            diff_ratio = len(set(modified) - set(original)) / max(len(original), 1)
            score += min(diff_ratio * 0.3, 0.3)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _sample_variations_horvitz_thompson(self, variations: List[Dict],
                                           problem_id: str, problem_text: str) -> List[Dict]:
        """
        Apply Horvitz-Thompson PPS sampling AFTER variation generation.

        This samples from already-generated variations, useful when:
        - Generation is fast but testing is slow
        - Want to test subset without losing generatable variations

        Returns sampled variations with tracking metadata for HT estimation.
        """
        if not variations:
            return variations

        total_variations = len(variations)
        variations_to_keep = int(total_variations * self.test_budget)

        if variations_to_keep >= total_variations:
            return variations

        self.logger.debug(f"\n      🎯 HT PPS SAMPLING (After Generation): Problem {problem_id}")
        self.logger.debug(f"         Total variations: {total_variations}")
        self.logger.debug(f"         Target sample: {variations_to_keep} ({int(self.test_budget*100)}%)")

        # Score each variation using fast heuristics
        items_for_sampling = []
        for idx, var in enumerate(variations):
            # Use existing scoring function
            score = self._score_variation_drift_likelihood_fast(var)
            var['variation_priority_score'] = score
            var['_var_idx'] = idx
            var['_problem_id'] = problem_id

            # Format for PPS: (id, data, score)
            var_id = f"{problem_id}_var_{idx}"
            items_for_sampling.append((var_id, var, score))

        # Perform PPS sampling
        sampled, inclusion_probs = self._horvitz_thompson_pps_sampling(
            items_for_sampling,
            variations_to_keep
        )

        # Store inclusion probabilities for later HT estimation
        if not hasattr(self, '_variation_inclusion_probs'):
            self._variation_inclusion_probs = {}
        self._variation_inclusion_probs.update(inclusion_probs)

        # Extract sampled variations
        sampled_variations = []
        for var_id, var, score in sampled:
            # Keep var_id for tracking
            var['_ht_var_id'] = var_id

            # Clean up temporary fields
            if '_var_idx' in var:
                del var['_var_idx']
            if '_problem_id' in var:
                del var['_problem_id']

            sampled_variations.append(var)

        # Track for statistics
        self.sampling_stats['variations_total'] += total_variations
        self.sampling_stats['variations_sampled'] += len(sampled_variations)

        self.logger.debug(f"      ✅ Selected {len(sampled_variations)} variations via PPS sampling")
        self.logger.debug(f"      ⏭️  Skipped {total_variations - len(sampled_variations)} variations")

        return sampled_variations

    def _filter_combinations_by_priority(self, all_combinations: Dict) -> Dict:
        """
        EARLY FILTERING: Sample combinations systematically BEFORE variation generation.

        Supports three methods:
        1. STRATIFIED SAMPLING: Ensures diversity across combination types
        2. HORVITZ-THOMPSON PPS: Unbiased estimates with confidence intervals
        3. NEYMAN + FACILITY LOCATION: Maximum coverage with multi-signal scoring

        Args:
            all_combinations: Dict mapping problem_idx to list of combinations

        Returns:
            Filtered dict with sampled combinations (or all if budget=1.0)
        """
        # Dispatch to appropriate sampling method
        if self.sampling_method == 'horvitz_thompson':
            return self._filter_combinations_horvitz_thompson(all_combinations)
        elif self.sampling_method == 'neyman_facility':
            return self._filter_combinations_neyman_facility(all_combinations)
        else:
            return self._filter_combinations_stratified(all_combinations)

    def _filter_combinations_stratified(self, all_combinations: Dict) -> Dict:
        """
        Stratified sampling of combinations.

        Strata:
        - Combination size: single-element vs multi-element
        - Domain composition: mono-domain vs cross-domain
        - Gradient bin: high (≥0.7) vs medium (0.4-0.7) vs low (<0.4)

        Within each stratum: prioritize by drift score
        """
        self.logger.debug(f"\n      🎯 STRATIFIED SAMPLING: Systematically selecting diverse combinations")

        total_combinations = sum(len(combos) for combos in all_combinations.values())
        combinations_to_keep = int(total_combinations * self.test_budget)

        self.logger.debug(f"         Total combinations: {total_combinations}")
        self.logger.debug(f"         Target sample size: {combinations_to_keep} ({int(self.test_budget*100)}%)")
        self.logger.debug(f"         Method: Stratified sampling (ensures diversity)")

        # Score all combinations and assign to strata
        scored_combinations = []

        for problem_idx, combinations in all_combinations.items():
            for combo in combinations:
                # Score based on candidate metadata
                score = self._score_combination_drift_likelihood(combo)
                combo['combination_priority_score'] = score

                # Extract stratum features
                size = combo.get('size', len(combo.get('candidates', [])))
                candidates = combo.get('candidates', [])
                domains = set(c.get('domain', '') for c in candidates if isinstance(c, dict))
                is_cross_domain = len(domains) >= 2

                # Gradient score bin (high/medium/low)
                if score >= 0.7:
                    gradient_bin = 'high'
                elif score >= 0.4:
                    gradient_bin = 'medium'
                else:
                    gradient_bin = 'low'

                # Create stratum key: (size_type, domain_type, gradient_bin)
                stratum = (
                    'single' if size == 1 else 'multi',
                    'cross' if is_cross_domain else 'mono',
                    gradient_bin
                )

                combo['_stratum'] = stratum
                scored_combinations.append((problem_idx, combo, score, stratum))

        self.logger.debug(f"      ✅ Scored {len(scored_combinations)} combinations (instant - zero cost!)")

        # Perform stratified sampling
        selected = self._stratified_sample_combinations(scored_combinations, combinations_to_keep)

        # Organize back into dict
        filtered = {}
        for problem_idx, combo, score, stratum in selected:
            if problem_idx not in filtered:
                filtered[problem_idx] = []
            # Remove temporary stratum key
            if '_stratum' in combo:
                del combo['_stratum']
            filtered[problem_idx].append(combo)

        skipped_count = len(scored_combinations) - len(selected)
        self.logger.debug(f"      ✅ Selected {len(selected)} diverse combinations via stratified sampling")
        self.logger.debug(f"      ⏭️  Skipped {skipped_count} combinations (no variations generated)")

        return filtered

    def _filter_combinations_horvitz_thompson(self, all_combinations: Dict) -> Dict:
        """
        Horvitz-Thompson PPS sampling of combinations.

        Advantages over stratified:
        - UNBIASED estimates of total drifts (E[estimate] = truth)
        - Confidence intervals (know our uncertainty)
        - Automatic weight adjustment for different drift probabilities
        - Theoretically optimal variance reduction

        Returns both sampled combinations AND inclusion probabilities for HT estimation.
        """
        self.logger.debug(f"\n      🎯 HORVITZ-THOMPSON PPS: Unbiased sampling with statistical estimation")

        total_combinations = sum(len(combos) for combos in all_combinations.values())
        combinations_to_keep = int(total_combinations * self.test_budget)

        self.logger.debug(f"         Total combinations: {total_combinations}")
        self.logger.debug(f"         Target sample size: {combinations_to_keep} ({int(self.test_budget*100)}%)")
        self.logger.debug(f"         Method: PPS (Probability Proportional to drift Score)")

        # Score all combinations and prepare for PPS sampling
        items_for_sampling = []
        combo_id = 0

        for problem_idx, combinations in all_combinations.items():
            for combo in combinations:
                # Score based on candidate metadata
                score = self._score_combination_drift_likelihood(combo)
                combo['combination_priority_score'] = score
                combo['_combo_id'] = combo_id
                combo['_problem_idx'] = problem_idx

                # Format for PPS: (id, data, score)
                items_for_sampling.append((combo_id, combo, score))
                combo_id += 1

        self.logger.debug(f"      ✅ Scored {len(items_for_sampling)} combinations (instant - zero cost!)")

        # Perform PPS sampling
        sampled, inclusion_probs = self._horvitz_thompson_pps_sampling(
            items_for_sampling,
            combinations_to_keep
        )

        # Store inclusion probabilities for later HT estimation
        self._combination_inclusion_probs = inclusion_probs
        self._combination_id_to_drifts = {}  # Will track drifts per combination

        # Organize back into dict
        filtered = {}
        for combo_id, combo, score in sampled:
            problem_idx = combo['_problem_idx']
            if problem_idx not in filtered:
                filtered[problem_idx] = []

            # Clean up temporary fields
            del combo['_combo_id']
            del combo['_problem_idx']

            # Keep combo_id for tracking
            combo['_ht_combo_id'] = combo_id  # For HT estimation later

            filtered[problem_idx].append(combo)

        skipped_count = len(items_for_sampling) - len(sampled)
        self.logger.debug(f"      ✅ Selected {len(sampled)} combinations via PPS sampling")
        self.logger.debug(f"      ⏭️  Skipped {skipped_count} combinations (no variations generated)")

        # Track for statistics
        self.sampling_stats['combinations_total'] = total_combinations
        self.sampling_stats['combinations_sampled'] = len(sampled)

        return filtered

    def _filter_combinations_neyman_facility(self, all_combinations: Dict) -> Dict:
        """
        Neyman Allocation + Facility Location sampling.

        This is the OPTIMAL approach for maximizing actual drifts found:

        Step 1: STRATIFICATION
          - Group combinations by (size, domain, gradient_bin)

        Step 2: NEYMAN ALLOCATION
          - Allocate budget proportional to variance within strata
          - n_h = n × (N_h × σ_h) / Σ(N_i × σ_i)
          - Allocates MORE samples to uncertain/high-variance strata

        Step 3: FACILITY LOCATION (within each stratum)
          - Greedy algorithm for maximum coverage
          - Selects combinations that maximize: drift_score × diversity
          - Guarantees (1-1/e) ≈ 63% of optimal coverage

        Advantages:
        - Robust to bad drift scores (hedges with uncertainty)
        - Guarantees coverage (facility location)
        - Multi-signal scoring (combines gradient + structural + diversity)
        - Optimal variance allocation (Neyman)
        """
        self.logger.debug(f"\n      🎯 NEYMAN + FACILITY LOCATION: Optimal coverage with multi-signal scoring")

        total_combinations = sum(len(combos) for combos in all_combinations.values())
        combinations_to_keep = int(total_combinations * self.test_budget)

        self.logger.debug(f"         Total combinations: {total_combinations}")
        self.logger.debug(f"         Target sample size: {combinations_to_keep} ({int(self.test_budget*100)}%)")
        self.logger.debug(f"         Method: Neyman allocation + Facility location greedy")

        # STEP 1: Stratification
        self.logger.debug(f"\n      📊 STEP 1: Stratification")

        strata = {}  # stratum_key -> list of combinations
        for problem_idx, combinations in all_combinations.items():
            for combo in combinations:
                # Extract stratum features
                size = combo.get('size', len(combo.get('candidates', [])))
                candidates = combo.get('candidates', [])
                domains = set(c.get('domain', '') for c in candidates if isinstance(c, dict))
                is_cross_domain = len(domains) >= 2

                # Get gradient score for binning
                gradient_scores = [
                    c.get('gradient_score', 0.0) for c in candidates
                    if isinstance(c, dict) and 'gradient_score' in c
                ]
                avg_gradient = sum(gradient_scores) / len(gradient_scores) if gradient_scores else 0.5

                # Gradient bin
                if avg_gradient >= 0.7:
                    gradient_bin = 'high'
                elif avg_gradient >= 0.4:
                    gradient_bin = 'medium'
                else:
                    gradient_bin = 'low'

                # Create stratum key
                stratum_key = (
                    'single' if size == 1 else 'multi',
                    'cross' if is_cross_domain else 'mono',
                    gradient_bin
                )

                if stratum_key not in strata:
                    strata[stratum_key] = []

                # Store with metadata
                combo['_stratum'] = stratum_key
                combo['_problem_idx'] = problem_idx
                strata[stratum_key].append(combo)

        self.logger.debug(f"         Created {len(strata)} strata:")
        for stratum_key, items in sorted(strata.items()):
            size_type, domain_type, gradient_bin = stratum_key
            self.logger.debug(f"           {size_type}-{domain_type}-{gradient_bin}: {len(items)} combinations")

        # STEP 2: Neyman Allocation
        self.logger.debug(f"\n      📐 STEP 2: Neyman Optimal Allocation")

        # Estimate variance for each stratum
        stratum_variances = {}
        for stratum_key, items in strata.items():
            variance = self._estimate_stratum_variance(items)
            stratum_variances[stratum_key] = variance

        # Neyman allocation: n_h = n × (N_h × σ_h) / Σ(N_i × σ_i)
        total_weighted = sum(
            len(items) * stratum_variances[key]
            for key, items in strata.items()
        )

        stratum_allocations = {}
        for stratum_key, items in strata.items():
            N_h = len(items)
            sigma_h = stratum_variances[stratum_key]

            if total_weighted > 0:
                # Neyman allocation
                allocation = int(combinations_to_keep * (N_h * sigma_h) / total_weighted)
                # But at least 1 if stratum is non-empty and budget allows
                allocation = max(1, min(allocation, N_h))
            else:
                # Fallback to proportional if no variance
                allocation = max(1, int(combinations_to_keep * N_h / total_combinations))

            stratum_allocations[stratum_key] = allocation

        # Normalize allocations to match budget
        total_allocated = sum(stratum_allocations.values())
        if total_allocated > combinations_to_keep:
            # Scale down proportionally
            scale = combinations_to_keep / total_allocated
            for key in stratum_allocations:
                stratum_allocations[key] = max(1, int(stratum_allocations[key] * scale))

        self.logger.debug(f"         Allocation (proportional to N_h × σ_h):")
        for stratum_key in sorted(strata.keys()):
            size_type, domain_type, gradient_bin = stratum_key
            N_h = len(strata[stratum_key])
            n_h = stratum_allocations[stratum_key]
            sigma_h = stratum_variances[stratum_key]
            self.logger.debug(f"           {size_type}-{domain_type}-{gradient_bin}: {n_h}/{N_h} "
                  f"(σ={sigma_h:.3f})")

        # STEP 3: Facility Location within each stratum
        self.logger.debug(f"\n      🏢 STEP 3: Facility Location (Greedy Coverage)")

        all_selected = []
        for stratum_key, items in strata.items():
            budget = stratum_allocations[stratum_key]

            self.logger.debug(f"\n         Stratum {stratum_key}: {budget} of {len(items)} combinations")

            # Run facility location greedy
            selected = self._facility_location_greedy(items, budget)
            all_selected.extend(selected)

        # Organize back into dict
        filtered = {}
        for combo in all_selected:
            problem_idx = combo['_problem_idx']
            if problem_idx not in filtered:
                filtered[problem_idx] = []

            # Clean up temporary fields
            if '_stratum' in combo:
                del combo['_stratum']
            if '_problem_idx' in combo:
                del combo['_problem_idx']

            filtered[problem_idx].append(combo)

        # Summary
        skipped_count = total_combinations - len(all_selected)
        self.logger.debug(f"\n      ✅ Selected {len(all_selected)} combinations via Neyman + Facility Location")
        self.logger.debug(f"      ⏭️  Skipped {skipped_count} combinations (no variations generated)")

        # Track for statistics
        self.sampling_stats['combinations_total'] = total_combinations
        self.sampling_stats['combinations_sampled'] = len(all_selected)

        # Print coverage analysis
        self.logger.debug(f"\n      📊 Coverage Analysis:")
        signal_types = {'gradient': [], 'structural': [], 'diversity': []}
        for combo in all_selected:
            if '_facility_scores' in combo:
                scores = combo['_facility_scores']
                signal_types['gradient'].append(scores['gradient_score'])
                signal_types['structural'].append(scores['structural_score'])
                signal_types['diversity'].append(scores['diversity_score'])

        if signal_types['gradient']:
            for signal_name, values in signal_types.items():
                avg = sum(values) / len(values)
                self.logger.debug(f"         {signal_name.capitalize()} score avg: {avg:.3f}")

        return filtered

    def _stratified_sample_combinations(self, scored_combinations: list, target_size: int) -> list:
        """
        Perform stratified sampling on combinations.

        Strategy:
        1. Group by stratum (size × domain × gradient)
        2. Allocate sample proportionally to stratum size
        3. Within each stratum, select top-scoring combinations
        4. Ensures we test diverse combination types

        Returns:
            List of selected (problem_idx, combo, score, stratum) tuples
        """
        # Group by stratum
        from collections import defaultdict
        strata = defaultdict(list)
        for item in scored_combinations:
            problem_idx, combo, score, stratum = item
            strata[stratum].append(item)

        self.logger.debug(f"\n      📊 Stratification results:")
        for stratum_key in sorted(strata.keys()):
            items = strata[stratum_key]
            size_type, domain_type, gradient_bin = stratum_key
            self.logger.debug(f"         {size_type}-element, {domain_type}-domain, {gradient_bin}-gradient: {len(items)} combinations")

        # Proportional allocation with minimum per stratum
        selected = []
        remaining_budget = target_size

        # Sort strata by priority (prefer high-gradient, multi-element, cross-domain)
        stratum_priority = []
        for stratum_key, items in strata.items():
            size_type, domain_type, gradient_bin = stratum_key
            priority = 0
            if gradient_bin == 'high': priority += 3
            elif gradient_bin == 'medium': priority += 2
            else: priority += 1

            if size_type == 'multi': priority += 1
            if domain_type == 'cross': priority += 1

            stratum_priority.append((priority, stratum_key, items))

        stratum_priority.sort(reverse=True)  # High priority first

        # Allocate proportionally
        total_items = len(scored_combinations)
        self.logger.debug(f"\n      🎯 Allocation strategy:")
        for priority, stratum_key, items in stratum_priority:
            if remaining_budget <= 0:
                break

            # Proportional allocation (but at least 1 if non-empty)
            stratum_size = len(items)
            allocation = max(1, int(target_size * (stratum_size / total_items)))
            allocation = min(allocation, remaining_budget, stratum_size)

            # Within stratum, select top-scoring
            items_sorted = sorted(items, key=lambda x: x[2], reverse=True)  # Sort by score
            stratum_selected = items_sorted[:allocation]
            selected.extend(stratum_selected)
            remaining_budget -= len(stratum_selected)

            size_type, domain_type, gradient_bin = stratum_key
            self.logger.debug(f"         → {size_type}-{domain_type}-{gradient_bin}: {len(stratum_selected)}/{len(items)} selected")

        # Fill remaining budget with top-scoring across all strata
        if remaining_budget > 0:
            all_unselected = [item for item in scored_combinations if item not in selected]
            all_unselected_sorted = sorted(all_unselected, key=lambda x: x[2], reverse=True)
            additional = all_unselected_sorted[:remaining_budget]
            selected.extend(additional)
            if additional:
                self.logger.debug(f"         → Backfill: {len(additional)} top-scoring combinations")

        return selected

    def _score_combination_drift_likelihood(self, combination: Dict) -> float:
        """
        Score how likely a combination is to cause drift (ZERO API calls!).

        Uses candidate metadata:
        - Gradient scores (CAGrad)
        - Combination size (multi-element = more impact)
        - Cross-domain (temporal+numeric = more drift)
        - Domain/topic information

        Returns:
            Drift likelihood score (0-1, higher = more likely to drift)
        """
        score = 0.5  # Default

        candidates = combination.get('candidates', [])
        if not candidates:
            return score

        # Factor 1: Gradient scores (if available from CAGrad)
        gradient_scores = [
            c.get('gradient_score', 0.0) for c in candidates
            if isinstance(c, dict) and 'gradient_score' in c
        ]
        if gradient_scores:
            # Use average gradient (all candidates contribute)
            avg_gradient = sum(gradient_scores) / len(gradient_scores)
            score = avg_gradient  # Gradient is already 0-1

            # Bonus for ALL high-gradient candidates
            if all(g > 0.7 for g in gradient_scores):
                score += 0.1  # All high-impact candidates = very likely to drift

        # Factor 2: Combination size (more candidates = more complex = more drift)
        size = combination.get('size', len(candidates))
        if size >= 4:
            score += 0.2
        elif size >= 2:
            score += 0.1

        # Factor 3: Cross-domain combinations (mixing domains = more fragile)
        domains = set(c.get('domain', '') for c in candidates if isinstance(c, dict))
        if len(domains) >= 2:
            score += 0.15  # Cross-domain bonus

        # Factor 4: Specific high-risk domains
        high_risk_domains = {'temporal', 'numeric', 'mathematical'}
        if any(c.get('domain', '') in high_risk_domains for c in candidates if isinstance(c, dict)):
            score += 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _multi_signal_score_combination(self, combination: Dict, selected_combinations: list = None) -> Dict:
        """
        Multi-signal scoring for combinations.

        Combines multiple signals to hedge against any single predictor being wrong:
        1. Gradient score (CAGrad) - if available
        2. Structural features (size, domain complexity)
        3. Diversity bonus (distance from already selected)

        Returns dict with:
            - 'total_score': Final weighted score
            - 'gradient_score': From CAGrad
            - 'structural_score': From features
            - 'diversity_score': Distance from selected
            - 'uncertainty': Prediction uncertainty
        """
        if selected_combinations is None:
            selected_combinations = []

        scores = {}

        # Signal 1: Gradient score (CAGrad)
        candidates = combination.get('candidates', [])
        gradient_scores = [
            c.get('gradient_score', 0.0) for c in candidates
            if isinstance(c, dict) and 'gradient_score' in c
        ]
        if gradient_scores:
            scores['gradient_score'] = sum(gradient_scores) / len(gradient_scores)
        else:
            scores['gradient_score'] = 0.5  # Neutral if no gradient

        # Signal 2: Structural features
        size = combination.get('size', len(candidates))
        domains = set(c.get('domain', '') for c in candidates if isinstance(c, dict))
        is_cross_domain = len(domains) >= 2

        structural = 0.5  # Base
        if size >= 4:
            structural += 0.3
        elif size >= 2:
            structural += 0.15

        if is_cross_domain:
            structural += 0.2

        high_risk_domains = {'temporal', 'numeric', 'mathematical'}
        if any(c.get('domain', '') in high_risk_domains for c in candidates if isinstance(c, dict)):
            structural += 0.1

        scores['structural_score'] = max(0.0, min(1.0, structural))

        # Signal 3: Diversity score (distance from already selected)
        if selected_combinations:
            # Measure feature distance from selected combinations
            min_distance = float('inf')

            for selected in selected_combinations:
                # Feature vector distance
                distance = self._combination_feature_distance(combination, selected)
                min_distance = min(min_distance, distance)

            # Normalize to [0, 1] - higher distance = more diverse = higher score
            scores['diversity_score'] = min(1.0, min_distance / 2.0)
        else:
            scores['diversity_score'] = 1.0  # First selection is maximally diverse

        # Signal 4: Uncertainty estimation
        # High uncertainty when signals disagree or score near boundary
        signal_values = [scores['gradient_score'], scores['structural_score']]
        signal_std = (sum((s - sum(signal_values)/len(signal_values))**2 for s in signal_values) / len(signal_values)) ** 0.5

        mean_score = sum(signal_values) / len(signal_values)
        boundary_uncertainty = 1.0 - abs(mean_score - 0.5) * 2  # High when near 0.5

        scores['uncertainty'] = signal_std + boundary_uncertainty * 0.5

        # Weighted combination
        weights = self.signal_weights
        total_score = (
            scores['gradient_score'] * weights.get('gradient', 0.4) +
            scores['structural_score'] * weights.get('structural', 0.3) +
            scores['diversity_score'] * weights.get('diversity', 0.3)
        )

        scores['total_score'] = max(0.0, min(1.0, total_score))

        return scores

    def _combination_feature_distance(self, combo1: Dict, combo2: Dict) -> float:
        """
        Calculate feature distance between two combinations.

        Features compared:
        - Size difference
        - Domain overlap (Jaccard distance)
        - Candidate overlap
        - Complexity difference
        """
        # Size distance
        size1 = combo1.get('size', len(combo1.get('candidates', [])))
        size2 = combo2.get('size', len(combo2.get('candidates', [])))
        size_dist = abs(size1 - size2) / max(size1, size2, 1)

        # Domain distance (Jaccard)
        candidates1 = combo1.get('candidates', [])
        candidates2 = combo2.get('candidates', [])

        domains1 = set(c.get('domain', '') for c in candidates1 if isinstance(c, dict))
        domains2 = set(c.get('domain', '') for c in candidates2 if isinstance(c, dict))

        if domains1 or domains2:
            jaccard = len(domains1 & domains2) / max(len(domains1 | domains2), 1)
            domain_dist = 1.0 - jaccard
        else:
            domain_dist = 0.0

        # Candidate text overlap (if available)
        texts1 = set(c.get('text', '') for c in candidates1 if isinstance(c, dict))
        texts2 = set(c.get('text', '') for c in candidates2 if isinstance(c, dict))

        if texts1 or texts2:
            text_jaccard = len(texts1 & texts2) / max(len(texts1 | texts2), 1)
            text_dist = 1.0 - text_jaccard
        else:
            text_dist = 0.0

        # Combined distance (average)
        total_dist = (size_dist + domain_dist + text_dist) / 3.0

        return total_dist

    def _estimate_stratum_variance(self, stratum_combinations: list) -> float:
        """
        Estimate prediction variance for a stratum.

        High variance indicates:
        - Scores disagree (gradient vs structural)
        - Rare combination type (small stratum)
        - Scores near decision boundary (0.4-0.6)
        - High within-stratum diversity

        Returns: Estimated variance (higher = more uncertain)
        """
        if not stratum_combinations:
            return 0.0

        # Score all combinations in stratum
        all_scores = []
        for combo in stratum_combinations:
            scores = self._multi_signal_score_combination(combo)
            all_scores.append(scores)

        # Factor 1: Score disagreement (std of uncertainties)
        uncertainties = [s['uncertainty'] for s in all_scores]
        score_disagreement = (sum((u - sum(uncertainties)/len(uncertainties))**2 for u in uncertainties) / len(uncertainties)) ** 0.5

        # Factor 2: Rarity penalty (small strata are uncertain)
        import math
        rarity = 1.0 / math.log(len(stratum_combinations) + 2)  # +2 to avoid log(1)=0

        # Factor 3: Boundary uncertainty (mean score near 0.5)
        mean_score = sum(s['total_score'] for s in all_scores) / len(all_scores)
        boundary = 1.0 - abs(mean_score - 0.5) * 2

        # Factor 4: Within-stratum variance (diversity of scores)
        score_values = [s['total_score'] for s in all_scores]
        within_variance = (sum((s - mean_score)**2 for s in score_values) / len(score_values)) ** 0.5

        # Weighted combination
        total_variance = (
            score_disagreement * 0.3 +
            rarity * 0.3 +
            boundary * 0.2 +
            within_variance * 0.2
        )

        return total_variance

    def _facility_location_greedy(self, candidates: list, budget: int) -> list:
        """
        Greedy facility location for maximum coverage.

        Iteratively selects combinations that maximize marginal gain:
            marginal_gain = drift_score × new_coverage

        Where:
        - drift_score: Multi-signal score (how likely to drift)
        - new_coverage: How much new feature space this covers

        Proven to achieve (1 - 1/e) ≈ 63% of optimal for submodular objectives.

        Args:
            candidates: List of combinations to select from
            budget: Number of combinations to select

        Returns:
            List of selected combinations
        """
        if not candidates or budget <= 0:
            return []

        selected = []

        self.logger.debug(f"      🏢 Facility Location: Selecting {budget} combinations for maximum coverage")

        for iteration in range(min(budget, len(candidates))):
            best_gain = -float('inf')
            best_combo = None
            best_scores = None

            # Evaluate marginal gain for each candidate
            for combo in candidates:
                if combo in selected:
                    continue

                # Multi-signal scoring with diversity
                scores = self._multi_signal_score_combination(combo, selected)

                # Marginal gain = score × diversity
                # (diversity is already in scores['diversity_score'])
                marginal_gain = scores['total_score']

                # Add coverage bonus: combinations in uncertain regions get boost
                coverage_bonus = scores['uncertainty'] * self.coverage_penalty
                total_gain = marginal_gain + coverage_bonus

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_combo = combo
                    best_scores = scores

            if best_combo is not None:
                selected.append(best_combo)

                # Store scores in combination for later analysis
                best_combo['_facility_scores'] = best_scores
                best_combo['_facility_gain'] = best_gain

                if (iteration + 1) % max(1, budget // 5) == 0:  # Print progress every 20%
                    self.logger.debug(f"         Selected {iteration + 1}/{budget}: gain={best_gain:.3f}, "
                          f"score={best_scores['total_score']:.3f}, diversity={best_scores['diversity_score']:.3f}")

        return selected

    def _horvitz_thompson_pps_sampling(self, items: list, target_size: int, score_key: str = 'score') -> tuple:
        """
        Horvitz-Thompson estimator with Probability Proportional to Size (PPS) sampling.

        This is the GOLD STANDARD for unbiased estimation with unequal probability sampling.

        Mathematical Foundation:
        - Sample with probability π_i ∝ drift_score_i
        - For sampled unit i: weight w_i = 1/π_i
        - HT estimate: θ̂ = Σ(y_i * w_i) where y_i = observed drifts from unit i
        - Variance: Var(θ̂) = Σ (1-π_i)/π_i * y_i^2  (Poisson PPS)
        - 95% CI: θ̂ ± 1.96 * sqrt(Var)

        Properties:
        - UNBIASED: E[θ̂] = θ (true total)
        - CONSISTENT: As n→∞, θ̂→θ
        - EFFICIENT: Lower variance than simple random sampling when size ∝ variable

        Args:
            items: List of (id, data, score) tuples
            target_size: Number of units to sample
            score_key: Key in data dict containing drift score (default: 'score')

        Returns:
            Tuple of (sampled_items, inclusion_probs_dict)
            - sampled_items: List of sampled (id, data, score) tuples
            - inclusion_probs: Dict mapping id → π_i (inclusion probability)
        """
        import random

        if not items or target_size <= 0:
            return [], {}

        # Extract scores
        total_score = sum(score for _, _, score in items)
        if total_score == 0:
            # Fallback to uniform sampling
            self.logger.debug(f"      ⚠️  All scores are zero, using uniform sampling")
            sampled = random.sample(items, min(target_size, len(items)))
            uniform_prob = target_size / len(items)
            inclusion_probs = {id_: uniform_prob for id_, _, _ in items}
            return sampled, inclusion_probs

        # Compute inclusion probabilities (PPS)
        # π_i = n * (score_i / Σ score_j)
        # But cap at 1.0 (can't have probability > 1)
        inclusion_probs = {}
        for id_, data, score in items:
            pi = min(1.0, target_size * (score / total_score))
            inclusion_probs[id_] = pi

        # Poisson PPS sampling: Include unit i with probability π_i
        sampled = []
        for id_, data, score in items:
            pi = inclusion_probs[id_]
            if random.random() < pi:
                sampled.append((id_, data, score))

        # Edge case: If we got too few samples (very low probabilities), add more
        if len(sampled) < min(3, target_size):
            self.logger.debug(f"      ⚠️  PPS sampling returned too few items ({len(sampled)}), supplementing")
            unsampled = [item for item in items if item not in sampled]
            if unsampled:
                supplement_size = min(target_size - len(sampled), len(unsampled))
                # Sample proportional to score
                weights = [score for _, _, score in unsampled]
                supplemental = random.choices(unsampled, weights=weights, k=supplement_size)
                sampled.extend(supplemental)

        self.logger.debug(f"      📊 PPS Sampling: {len(sampled)}/{len(items)} items selected")
        self.logger.debug(f"         Inclusion probability range: [{min(inclusion_probs.values()):.3f}, {max(inclusion_probs.values()):.3f}]")

        return sampled, inclusion_probs

    def _horvitz_thompson_estimate(self, sampled_results: list, inclusion_probs: dict,
                                   total_population: int) -> tuple:
        """
        Compute Horvitz-Thompson estimate with confidence intervals.

        Args:
            sampled_results: List of (id, num_drifts) tuples from tested samples
            inclusion_probs: Dict mapping id → π_i (from PPS sampling)
            total_population: Total number of units in population (for context)

        Returns:
            Tuple of (estimated_total, variance, ci_lower, ci_upper)
        """
        import math

        if not sampled_results:
            return 0.0, 0.0, 0.0, 0.0

        # HT estimator: θ̂ = Σ(y_i / π_i)
        estimated_total = 0.0
        variance = 0.0

        for id_, num_drifts in sampled_results:
            pi = inclusion_probs.get(id_, 1.0)
            if pi > 0:
                weight = 1.0 / pi
                estimated_total += num_drifts * weight

                # Variance contribution (Poisson PPS formula)
                # Var = Σ (1-π_i)/π_i * y_i^2
                variance += ((1 - pi) / pi) * (num_drifts ** 2)

        # Confidence interval
        std_error = math.sqrt(variance) if variance > 0 else 0.0

        # For 95% CI: ± 1.96 * SE
        # For 99% CI: ± 2.576 * SE
        z_score = 1.96 if self.ht_confidence_level == 0.95 else 2.576

        ci_lower = max(0, estimated_total - z_score * std_error)
        ci_upper = estimated_total + z_score * std_error

        self.logger.debug(f"\n      📈 HORVITZ-THOMPSON ESTIMATION:")
        self.logger.debug(f"         Sampled: {len(sampled_results)}/{total_population} units")
        self.logger.debug(f"         Drifts found: {sum(num for _, num in sampled_results)}")
        self.logger.debug(f"         Estimated total drifts: {estimated_total:.1f}")
        self.logger.debug(f"         Standard error: {std_error:.2f}")
        self.logger.debug(f"         {int(self.ht_confidence_level*100)}% Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}]")

        return estimated_total, variance, ci_lower, ci_upper

    def _print_ht_estimation_summary(self):
        """
        Print comprehensive Horvitz-Thompson estimation summary and comparison.

        This provides:
        1. Sampling statistics (how many combinations/variations sampled)
        2. Comparison of sampling strategies (before vs after vs both)
        3. Statistical interpretation and recommendations
        """
        self.logger.debug(f"\n{'='*80}")
        self.logger.debug(f"📊 HORVITZ-THOMPSON STATISTICAL SAMPLING SUMMARY")
        self.logger.debug(f"{'='*80}")

        # Sampling configuration
        self.logger.debug(f"\n⚙️  Configuration:")
        self.logger.debug(f"   Sampling Method: Horvitz-Thompson PPS (Probability Proportional to Size)")
        self.logger.debug(f"   Sampling Stage: {self.ht_sampling_stage}")
        self.logger.debug(f"   Test Budget: {int(self.test_budget * 100)}%")
        self.logger.debug(f"   Confidence Level: {int(self.ht_confidence_level * 100)}%")

        # Sampling statistics
        self.logger.debug(f"\n📈 Sampling Statistics:")
        if self.sampling_stats['combinations_total'] > 0:
            self.logger.debug(f"   Combinations:")
            self.logger.debug(f"     Total: {self.sampling_stats['combinations_total']}")
            self.logger.debug(f"     Sampled: {self.sampling_stats['combinations_sampled']}")
            combo_pct = (self.sampling_stats['combinations_sampled'] /
                        self.sampling_stats['combinations_total'] * 100)
            self.logger.debug(f"     Sampling Rate: {combo_pct:.1f}%")

        if self.sampling_stats['variations_total'] > 0:
            self.logger.debug(f"   Variations:")
            self.logger.debug(f"     Total: {self.sampling_stats['variations_total']}")
            self.logger.debug(f"     Sampled: {self.sampling_stats['variations_sampled']}")
            var_pct = (self.sampling_stats['variations_sampled'] /
                      self.sampling_stats['variations_total'] * 100)
            self.logger.debug(f"     Sampling Rate: {var_pct:.1f}%")

        # Speedup calculation
        self.logger.debug(f"\n⚡ Speedup Analysis:")
        if self.ht_sampling_stage == 'before':
            # Saved on both generation and testing
            combo_saved = self.sampling_stats['combinations_total'] - self.sampling_stats['combinations_sampled']
            vars_per_combo = 3  # Typical
            total_saved_generation = combo_saved * vars_per_combo
            total_saved_testing = combo_saved * vars_per_combo
            self.logger.debug(f"   Strategy: Sample BEFORE variation generation")
            self.logger.debug(f"   Skipped Combinations: {combo_saved}")
            self.logger.debug(f"   Saved Variation Generations: ~{total_saved_generation} API calls")
            self.logger.debug(f"   Saved Variation Tests: ~{total_saved_testing} API calls")
            self.logger.debug(f"   Total API Savings: ~{total_saved_generation + total_saved_testing} calls")

        elif self.ht_sampling_stage == 'after':
            var_saved = self.sampling_stats['variations_total'] - self.sampling_stats['variations_sampled']
            self.logger.debug(f"   Strategy: Sample AFTER variation generation")
            self.logger.debug(f"   Skipped Variations: {var_saved}")
            self.logger.debug(f"   Saved Variation Tests: ~{var_saved} API calls")
            self.logger.debug(f"   NOTE: Generation cost already paid, only test cost saved")

        elif self.ht_sampling_stage == 'both':
            self.logger.debug(f"   Strategy: Two-level sampling (BEFORE + AFTER)")
            self.logger.debug(f"   WARNING: Two-level sampling can compound quality loss")
            self.logger.debug(f"   Recommendation: Use 'before' for most cases")

        # Statistical properties
        self.logger.debug(f"\n📐 Statistical Properties:")
        self.logger.debug(f"   ✅ UNBIASED: E[estimate] = true total drifts")
        self.logger.debug(f"   ✅ CONSISTENT: Estimate → truth as sample size → ∞")
        self.logger.debug(f"   ✅ EFFICIENT: Variance < simple random sampling")
        self.logger.debug(f"   ✅ INTERPRETABLE: Confidence intervals quantify uncertainty")

        # Comparison with stratified sampling
        self.logger.debug(f"\n🔄 Comparison: HT vs Stratified Sampling")
        self.logger.debug(f"   {'Strategy':<20} {'Pros':<35} {'Cons':<25}")
        self.logger.debug(f"   {'-'*80}")
        self.logger.debug(f"   {'Stratified':<20} {'Ensures diversity':<35} {'No stat. guarantees':<25}")
        self.logger.debug(f"   {'':<20} {'Easy to understand':<35} {'May miss rare drifts':<25}")
        self.logger.debug(f"   {'':<20} {'Good for exploration':<35} {'':<25}")
        self.logger.debug(f"   {'-'*80}")
        self.logger.debug(f"   {'Horvitz-Thompson':<20} {'Unbiased estimates':<35} {'More complex':<25}")
        self.logger.debug(f"   {'':<20} {'Confidence intervals':<35} {'Needs good scores':<25}")
        self.logger.debug(f"   {'':<20} {'Optimal for inference':<35} {'':<25}")

        # Recommendations
        self.logger.debug(f"\n💡 Recommendations:")
        self.logger.debug(f"   When to use Stratified:")
        self.logger.debug(f"     - Early exploration (understand drift types)")
        self.logger.debug(f"     - Need guaranteed diversity across types")
        self.logger.debug(f"     - Don't need precise total drift count")
        self.logger.debug(f"\n   When to use Horvitz-Thompson:")
        self.logger.debug(f"     - Need unbiased estimates of total drifts")
        self.logger.debug(f"     - Want confidence intervals (know your uncertainty)")
        self.logger.debug(f"     - Have good drift scores (gradient scores from CAGrad)")
        self.logger.debug(f"     - Publishing results (statistical rigor)")

        self.logger.debug(f"\n   Optimal Stage (BEFORE vs AFTER):")
        self.logger.debug(f"     BEFORE: ✅ Recommended - saves both generation AND testing")
        self.logger.debug(f"     AFTER:  Use only if variations are cheap to generate")
        self.logger.debug(f"     BOTH:   ⚠️  Not recommended - compounds uncertainty")

        self.logger.debug(f"\n{'='*80}\n")

    def filter_variations_by_priority(self, variations_by_problem: Dict,
                                      baseline_answers: Dict,
                                      model_client,
                                      test_budget_pct: float = 1.0) -> Dict:
        """
        Filter variations using prioritized testing strategy.

        Strategy:
        1. Score ALL variations with logprobs (fast - no generation)
        2. Prioritize by drift likelihood
        3. Only fully test top N% (configurable budget)
        4. Mark untested variations for potential future testing

        Args:
            variations_by_problem: Dict mapping problem_idx to list of variations
            baseline_answers: Dict mapping problem_idx to ground truth answer
            model_client: Model client with logprobs support
            test_budget_pct: Percentage of variations to fully test (0.0-1.0)

        Returns:
            Dict mapping problem_idx to prioritized variations (with drift_priority scores)
        """
        if test_budget_pct >= 1.0:
            # No filtering - test everything
            self.logger.debug(f"      ℹ️  Prioritized testing disabled (budget=100%) - testing all variations")
            return variations_by_problem

        self.logger.debug(f"\n      🎯 PRIORITIZED TESTING: Scoring variations to identify high-drift candidates")
        self.logger.debug(f"         Test budget: {int(test_budget_pct*100)}% of variations")

        prioritized = {}
        total_variations = sum(len(vars) for vars in variations_by_problem.values())
        total_to_test = int(total_variations * test_budget_pct)

        self.logger.debug(f"         Total variations: {total_variations}")
        self.logger.debug(f"         Will test: {total_to_test} ({int(test_budget_pct*100)}%)")
        self.logger.debug(f"         Speedup: {1/test_budget_pct:.1f}x")

        # Step 1: Score all variations (SMART - use existing CAGrad scores!)
        self.logger.debug(f"\n      📊 Phase 1: Computing drift priority scores (zero-cost)...")
        scored_variations = []
        cagrad_scores_used = 0
        heuristic_scores_used = 0

        for problem_idx, variations in variations_by_problem.items():
            for var in variations:
                # SMART: Check if variation has CAGrad gradient scores
                # These ALREADY measure how much the candidate affects P(correct_answer)!
                existing_gradient = None

                # Check candidates for gradient scores
                candidates = var.get('candidates', [])
                if candidates:
                    candidate_gradients = [
                        c.get('gradient_score', 0.0) for c in candidates
                        if isinstance(c, dict) and 'gradient_score' in c
                    ]
                    if candidate_gradients:
                        # Use max gradient (most impactful candidate)
                        existing_gradient = max(candidate_gradients)
                        cagrad_scores_used += 1

                # Compute drift score
                if existing_gradient is not None and existing_gradient > 0:
                    # ZERO COST: Use existing CAGrad gradient score!
                    drift_score = existing_gradient
                else:
                    # Fast heuristic for non-CAGrad variations
                    drift_score = self._compute_heuristic_drift_score(var)
                    heuristic_scores_used += 1

                # Add score to variation
                var['drift_priority_score'] = drift_score
                var['priority_testing_applied'] = True

                scored_variations.append((problem_idx, var, drift_score))

        if cagrad_scores_used > 0:
            self.logger.debug(f"      ✅ Used {cagrad_scores_used} existing CAGrad scores (ZERO API calls!)")
        if heuristic_scores_used > 0:
            self.logger.debug(f"      ✅ Computed {heuristic_scores_used} fast heuristic scores (instant)")
        self.logger.debug(f"      📊 Total scored: {len(scored_variations)} variations")

        # Step 2: Sort by drift likelihood (high to low)
        scored_variations.sort(key=lambda x: x[2], reverse=True)

        # Step 3: Select top N% for testing
        selected_for_testing = scored_variations[:total_to_test]
        skipped = scored_variations[total_to_test:]

        self.logger.debug(f"\n      🎯 Phase 2: Prioritization results:")
        self.logger.debug(f"         High priority (will test): {len(selected_for_testing)}")
        self.logger.debug(f"         Low priority (skip for now): {len(skipped)}")

        if selected_for_testing:
            top_score = selected_for_testing[0][2]
            bottom_score = selected_for_testing[-1][2]
            self.logger.debug(f"         Score range for testing: {bottom_score:.3f} - {top_score:.3f}")

        # Step 4: Organize selected variations by problem
        for problem_idx, var, score in selected_for_testing:
            var['selected_for_testing'] = True
            if problem_idx not in prioritized:
                prioritized[problem_idx] = []
            prioritized[problem_idx].append(var)

        # Step 5: Mark skipped variations (for potential future testing)
        for problem_idx, var, score in skipped:
            var['selected_for_testing'] = False
            var['skip_reason'] = 'low_drift_priority'
            # Don't add to prioritized dict - these won't be tested

        return prioritized

    # ═══════════════════════════════════════════════════════════════════════════════
    # DEPRECATED METHODS - OLD BUCKET+SUBSTITUTION APPROACH
    # ═══════════════════════════════════════════════════════════════════════════════
    # These methods implement the OLD approach (bucket generation + substitution).
    # They are kept for reference but should be removed in future cleanup.
    # The NEW approach (direct variation) is implemented above.
    # ═══════════════════════════════════════════════════════════════════════════════

    def _batch_candidate_buckets_cross_problem(self, problem_data: List[Dict],
                                             valid_problems: Dict, engine) -> Dict:
        """
        DEPRECATED: Batch candidate bucket generation across ALL problems and candidates.

        This method is part of the OLD bucket+substitution approach.
        The NEW direct variation approach doesn't need bucket generation.

        ⚠️ FLAGGED FOR REMOVAL - Do not use for new code.
        """

        # Collect ALL candidate bucket requests across ALL problems
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track (problem_idx, candidate_idx) for each request

        total_requests = 0
        for problem_idx, candidates in valid_problems.items():
            problem_text = problem_data[problem_idx]['problem_text']

            for candidate_idx, candidate in enumerate(candidates):
                # Generate the same prompts as the original _generate_candidate_buckets
                highlighted_question = problem_text.replace(candidate['text'], f"→→→{candidate['text']}←←←")
                variants_per_candidate = 4

                system_prompt = f"""You are an expert at generating contextually-appropriate alternative representations that will be substituted in place of specific elements in questions.

CRITICAL TASK: Generate {variants_per_candidate} different alternatives for the marked element that will be substituted IN-PLACE in the original question.

🚨 SUBSTITUTION TEST: Each variant must make sense when replacing the original element in context.

ESSENTIAL REQUIREMENTS:
1. ✅ PRESERVE exact meaning, numerical value, and format appropriateness
2. ✅ ENSURE the substituted question remains grammatically correct and coherent
3. ✅ MAINTAIN the same answer and logical structure
4. ✅ RESPECT compositional units (don't break apart formats like "H:M:S", "X hours Y minutes")
5. ✅ Use appropriate register for the question context

DIVERSITY & COMPLEXITY GUIDELINES:
🎯 CREATE meaningful, diverse alternatives (not trivial changes)
🎯 Each variant should be SUBSTANTIALLY DIFFERENT from others in:
   - Word choice and phrasing style
   - Level of formality (formal/informal/technical)
   - Complexity level (simple/detailed/descriptive)
   - Representation format (numeric/textual/abbreviated)

OUTPUT FORMAT: Return exactly {variants_per_candidate} variants, one per line:
<variants>
variant 1
variant 2
variant 3
variant 4
</variants>"""

                user_prompt = f"""ORIGINAL QUESTION: {problem_text}
HIGHLIGHTED CONTEXT: {highlighted_question}
ELEMENT TO VARY: "{candidate['text']}"
ELEMENT TYPE: {candidate['domain']} - {candidate['topic']}

🎯 TASK: Generate {variants_per_candidate} alternatives that can be substituted IN-PLACE without changing:
- The question's core meaning
- The mathematical/logical structure
- The expected answer type
- Grammatical correctness

Generate {variants_per_candidate} high-quality alternatives optimized for in-place substitution."""

                all_system_prompts.append(system_prompt)
                all_user_prompts.append(user_prompt)
                request_mapping.append((problem_idx, candidate_idx, candidate))
                total_requests += 1

        self.logger.debug(f"          📦 Batching {total_requests} candidate bucket requests across {len(valid_problems)} problems...")

        # Execute massive batch call
        all_candidate_buckets = {}
        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 10 == 0:
                        self.logger.debug(f"            Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses and organize by problem
            successful_buckets = 0
            for i, (response, (problem_idx, candidate_idx, candidate)) in enumerate(zip(responses, request_mapping)):
                # Extract variants using same logic as original
                import re
                variants_match = re.search(r'<variants>(.*?)</variants>', response, re.DOTALL)
                if variants_match:
                    variant_lines = variants_match.group(1).strip().split('\n')
                    raw_variants = [line.strip() for line in variant_lines if line.strip()]

                    if raw_variants:
                        # VALIDATE variants using exact same logic as individual processing
                        validated_variants = engine._validate_candidate_variants(candidate['text'], raw_variants)

                        if validated_variants and len(validated_variants) >= 2:
                            if problem_idx not in all_candidate_buckets:
                                all_candidate_buckets[problem_idx] = {}
                            all_candidate_buckets[problem_idx][candidate['text']] = validated_variants
                            successful_buckets += 1
                        else:
                            # Use fallback variants (same as individual processing)
                            fallback_variants = engine._generate_fallback_variants(candidate)
                            if problem_idx not in all_candidate_buckets:
                                all_candidate_buckets[problem_idx] = {}
                            all_candidate_buckets[problem_idx][candidate['text']] = fallback_variants
                            successful_buckets += 1

            self.logger.debug(f"          ✅ Candidate bucket batch: {successful_buckets}/{total_requests} successful")

        except Exception as e:
            self.logger.debug(f"          ❌ Candidate bucket batch failed: {e}")

        return all_candidate_buckets

    def _batch_combination_selection_cross_problem(self, problem_data: List[Dict],
                                                 valid_problems: Dict, engine) -> Dict:
        """Batch combination selection across ALL problems with FRAGMENT AWARENESS."""

        # Collect ALL combination selection requests
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track problem_idx for each request

        for problem_idx, candidates in valid_problems.items():
            problem_text = problem_data[problem_idx]['problem_text']
            max_combinations = self.max_combinations  # User configurable

            # Create candidate list (same as original logic)
            # NOTE: Fragments are already in candidates list (added during detection)
            candidate_list = []
            for i, cand in enumerate(candidates):
                candidate_list.append(f"[{i}] '{cand['text']}' (domain:{cand['domain']}, topic:{cand['topic']})")

            system_prompt = f"""You are an expert at selecting diverse combinations of candidates for creating varied question transformations.
TASK: Select EXACTLY {max_combinations} UNIQUE, DIVERSE combinations with VARIED SIZES.

🚨 CRITICAL UNIQUENESS REQUIREMENT:
- NEVER REPEAT A COMBINATION - each must be COMPLETELY UNIQUE
- [0,1] and [1,0] are THE SAME (order doesn't matter) - only output ONE
- [0,2] and [0,2] are duplicates - only output ONCE
- Before adding a combination, check if you've already included it
- STOP immediately after outputting {max_combinations} unique combinations

DIVERSITY REQUIREMENTS:
1. Mix of combination sizes: EXPLOIT MAXIMUM POSSIBLE based on available candidates
2. Minimize candidate overlap between combinations
3. Include different domain mixtures (math+temporal, nl+math, etc.)
4. Focus on testing different cognitive capabilities
5. Ensure each combination transforms meaningfully different aspects

SELECTION STRATEGY:
- START with LARGEST possible combinations (use as many candidates as feasible)
- Create combinations that span maximum diversity of domains/topics
- Generate EXACTLY {max_combinations} combinations, no more, no less

CRITICAL OUTPUT FORMAT: Use this ULTRA-COMPACT format to minimize tokens:
0,2,4,6|1,3|5,7,8|1,4
Rules:
- Each combination = comma-separated indices
- Different combinations = separated by |
- 0-based indexing
- NO JSON, NO brackets, NO extra text
- Example: "0,1,3|2,4|0,5,6,7" means 3 combinations: [0,1,3], [2,4], [0,5,6,7]"""

            user_prompt = f"""Problem: {problem_text}

Available candidates ({len(candidates)} total):
{chr(10).join(candidate_list)}

Select diverse combinations using the candidate indices [0] through [{len(candidates)-1}].

STRATEGY:
- You have {len(candidates)} candidates available
- Create combinations that use different sizes and domains
- If you have 10+ candidates, consider 6-8 way combinations
- If you have 7-9 candidates, consider 4-6 way combinations
- If you have 4-6 candidates, consider 3-4 way combinations
- Always include some smaller combinations for targeted testing

Respond with indices only in format: 0,2,4|1,3|5,7

Select diverse combinations (up to {max_combinations}) that maximize candidate utilization and vary significantly in size.

CRITICAL: You must respond with ONLY combination indices in ultra-compact format.
Example: 0,2,4|1,3|5,7
DO NOT include any other text, numbers, or explanations. Only the index combinations."""

            all_system_prompts.append(system_prompt)
            all_user_prompts.append(user_prompt)
            request_mapping.append(problem_idx)

        self.logger.debug(f"          📦 Batching {len(all_system_prompts)} MODEL-GUIDED combination selection requests...")

        # Execute batch call - MODEL WILL INTELLIGENTLY SELECT DIVERSE COMBINATIONS
        all_selected_combinations = {}
        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 5 == 0:
                        self.logger.debug(f"            Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses
            successful_selections = 0
            for i, (response, problem_idx) in enumerate(zip(responses, request_mapping)):
                # Use EXACT same logic as engine._model_select_diverse_combinations
                import re
                selected_combinations = []
                candidates = list(valid_problems[problem_idx])
                self.logger.debug(f"            🤖 Model selecting combinations for problem {problem_idx + 1} from {len(candidates)} candidates...")

                # Clean response - remove common wrapper text
                clean_response = response.strip()
                for unwrap_pattern in [r'```[^`]*```', r'<[^>]*>', r'"([^"]*)"']:
                    match = re.search(unwrap_pattern, clean_response)
                    if match:
                        clean_response = match.group(1) if match.groups() else match.group(0)
                        break

                # Extract compact format: look for pattern like "0,1,3|2,4|5,6,7"
                compact_patterns = [
                    r'(\d+(?:,\d+)*(?:\|\d+(?:,\d+)*)*)',  # Primary pattern
                    r'([0-9,|]+)',                          # Fallback: any digits, commas, pipes
                ]
                combinations_text = None
                for pattern in compact_patterns:
                    match = re.search(pattern, clean_response)
                    if match:
                        combinations_text = match.group(1)
                        break

                if combinations_text:
                    # try:
                        # Split by | to get individual combinations
                        combo_strings = combinations_text.split('|')
                        seen_combinations = set()  # Track seen combinations to avoid duplicates

                        for combo_str in combo_strings:
                            combo_str = combo_str.strip()
                            if not combo_str:
                                continue
                            try:
                                # Parse comma-separated indices
                                indices = [int(x.strip()) for x in combo_str.split(',')]

                                # Validate indices
                                if not all(0 <= idx < len(candidates) for idx in indices):
                                    continue

                                # Get candidates for these indices
                                combo_candidates = [candidates[i] for i in indices]

                                # Validate candidates have required fields
                                if not all(c and 'text' in c for c in combo_candidates):
                                    continue

                                # ATOMIC-LEVEL DEDUPLICATION: Remove duplicate candidate texts within combination
                                # E.g., ["9:00 AM", "9:00 AM", "9:00", "2"] → ["9:00 AM", "9:00", "2"]
                                seen_texts = set()
                                unique_combo_candidates = []
                                for cand in combo_candidates:
                                    if cand['text'] not in seen_texts:
                                        seen_texts.add(cand['text'])
                                        unique_combo_candidates.append(cand)
                                combo_candidates = unique_combo_candidates

                                # After deduplication, check if we still have candidates
                                if not combo_candidates:
                                    continue

                                # COMBINATION-LEVEL DEDUPLICATION: Skip if this exact set of texts already exists
                                # Use frozenset for order-independent comparison
                                combo_key = frozenset(c['text'] for c in combo_candidates)
                                if combo_key in seen_combinations:
                                    continue  # Skip duplicate combination
                                seen_combinations.add(combo_key)

                                # Final integrity check: size matches actual count
                                actual_size = len(combo_candidates)
                                selected_combinations.append({
                                    'candidates': combo_candidates,
                                    'reason': f'Model-selected diverse {actual_size}-way combination',
                                    'size': actual_size,  # Guaranteed to match
                                    'priority': 100 - actual_size
                                })
                            except (ValueError, IndexError):
                                continue


                # Check if we have successful combinations
                if selected_combinations:
                    original_count = len(combo_strings)
                    unique_count = len(selected_combinations)
                    duplicates_removed = original_count - unique_count

                    result = selected_combinations[:self.max_combinations]
                    all_selected_combinations[problem_idx] = result
                    successful_selections += 1
                    sizes = [c['size'] for c in result]

                    if duplicates_removed > 0:
                        self.logger.debug(f"            ✅ Model selected {len(result)} diverse combinations (sizes: {sizes}) [{duplicates_removed} duplicates removed]")
                    else:
                        self.logger.debug(f"            ✅ Model selected {len(result)} diverse combinations (sizes: {sizes})")
                else:
                    self.logger.debug(f"            ⚠️ No valid combinations extracted for problem {problem_idx + 1}")

            self.logger.debug(f"          ✅ Combination selection batch: {successful_selections}/{len(request_mapping)} successful")

        except Exception as e:
            self.logger.debug(f"          ❌ Combination selection batch failed: {e}")

        return all_selected_combinations

    def _batch_final_variations_cross_problem(self, problem_data: List[Dict], valid_problems: Dict,
                                            all_selected_combinations: Dict, all_candidate_buckets: Dict,
                                            engine) -> Dict:
        """
        DEPRECATED: Batch final variation creation across ALL problems in ONE massive batch call.

        This method is part of the OLD bucket+substitution approach.
        The NEW direct variation approach generates complete variations directly.

        ⚠️ FLAGGED FOR REMOVAL - Do not use for new code.
        """

        import random
        import re
        from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question

        # Collect ALL final variation prompts from ALL problems
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track (problem_idx, combo_data, selected_transformations, transformation_details)

        total_requests = 0
        for problem_idx in valid_problems.keys():
            if problem_idx not in all_selected_combinations or problem_idx not in all_candidate_buckets:
                continue

            problem_text = problem_data[problem_idx]['problem_text']
            selected_combinations = all_selected_combinations[problem_idx]
            candidate_buckets = all_candidate_buckets[problem_idx]

            # Generate prompts for each combination (EXACT same logic as engine._create_bucket_variations_batched)
            for combo_data in selected_combinations:
                candidates = combo_data['candidates']

                # PRESERVE: Random selection from buckets for this combination (line 209 in engine)
                selected_transformations = {}
                transformation_details = []

                for candidate in candidates:
                    candidate_text = candidate['text']
                    if candidate_text in candidate_buckets and candidate_buckets[candidate_text]:
                        selected_transformation = random.choice(candidate_buckets[candidate_text])
                        selected_transformations[candidate_text] = selected_transformation
                        transformation_details.append(f"'{candidate_text}' → '{selected_transformation}'")
                    else:
                        selected_transformations[candidate_text] = candidate_text
                        transformation_details.append(f"'{candidate_text}' → '{candidate_text}' (fallback)")

                # PRESERVE: Static guidance from engine (line 217 in engine)
                static_guidance = engine._get_static_guidance_for_combination(candidates)

                # EXACT same system prompt as engine (updated with examples and identity transformation prohibition)
                system_prompt = f"""You are an expert at creating meaningful question variations by transforming multiple candidates simultaneously.

CRITICAL REQUIREMENTS:
1. PRESERVE exact numerical answer and question intent
2. USE EXACTLY the specified transformations provided
3. TRANSFORM all specified candidates using the given transformations
4. ENSURE mathematical/logical validity
5. Use PLAIN TEXT only - no formatting

{static_guidance}

ABSOLUTELY FORBIDDEN:
✗ Identity transformations: "apple" → "apple" (same-to-same is NOT a transformation)
✗ Changing core question logic or numerical answer
✗ Modifying transformations that were provided
✗ Changing problem structure or intent

QUALITY EXAMPLES OF COMPLETE TRANSFORMATIONS:

Example 1 (Math word problem):
Original: "A car travels 60 miles in 2 hours. What is its speed?"
Transformations: "60" → "sixty" | "miles" → "mi" | "2 hours" → "120 minutes"
✓ GOOD: "A car travels sixty mi in 120 minutes. What is its speed?"
✗ BAD: "A vehicle goes 60 miles in 2 hours. What's the velocity?" (changed core logic)

Example 2 (Temporal problem):
Original: "Meeting starts at 3:00 PM and lasts 90 minutes. When does it end?"
Transformations: "3:00 PM" → "fifteen hundred hours" | "90 minutes" → "one and a half hours"
✓ GOOD: "Meeting starts at fifteen hundred hours and lasts one and a half hours. When does it end?"
✗ BAD: "Session begins at 3:00 PM lasting 90 minutes. What's the conclusion time?" (changed wording beyond transformations)

Example 3 (Mixed domains):
Original: "John walks 3/4 of a mile in 15 minutes. How fast does he walk?"
Transformations: "John" → "a person" | "walks" → "travels on foot" | "3/4" → "three quarters"
✓ GOOD: "A person travels on foot three quarters of a mile in 15 minutes. How fast does he walk?"
✗ BAD: "John walks 3/4 mile in 15 mins. What's his pace?" (didn't use exact transformations)

OUTPUT FORMAT: Return ONLY the modified question wrapped in <question> tags.
Example: <question>Your modified question here</question>"""

                # EXACT same user prompt as engine (lines 237-244 in engine)
                user_prompt = f"""Original: {problem_text}

Apply these EXACT transformations:
{chr(10).join(transformation_details)}

Combination: {combo_data['reason']}

Create ONE variation using EXACTLY the specified transformations while preserving exact answer and intent."""

                all_system_prompts.append(system_prompt)
                all_user_prompts.append(user_prompt)
                request_mapping.append((problem_idx, combo_data, selected_transformations, transformation_details))
                total_requests += 1

        if not all_system_prompts:
            self.logger.debug(f"          ℹ️  No final variation requests to batch")
            return {}

        self.logger.debug(f"          📦 Batching {total_requests} final variation requests across {len(valid_problems)} problems...")

        # Execute ONE massive batch call
        combination_variations_map = {}
        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                # Fallback for non-batching clients
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 50 == 0:
                        self.logger.debug(f"            Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses and map back to problems (EXACT same logic as engine lines 264-290)
            successful_variations = 0
            for i, (response, (problem_idx, combo_data, selected_transformations, transformation_details)) in enumerate(
                zip(responses, request_mapping)):

                candidates = combo_data['candidates']

                # EXACT same response parsing as engine (lines 270-287)
                question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
                if question_match:
                    cleaned = clean_model_response(question_match.group(1).strip())

                    if cleaned and is_valid_question(cleaned):
                        variation = {
                            'original_problem': problem_data[problem_idx]['problem_text'],
                            'modified_problem': cleaned,
                            'transformation_type': f'bucket_combination_{combo_data["size"]}way',
                            'candidates_transformed': [c['text'] for c in candidates],
                            'selected_variants': selected_transformations,
                            'transformation_details': transformation_details,
                            'combination_size': combo_data['size'],
                            'domains_involved': list(set(c['domain'] for c in combo_data['candidates'])),
                            'cross_domain': len(set(c['domain'] for c in combo_data['candidates'])) > 1,
                            'confidence': 'model_generated',
                            'generation_method': 'cross_problem_batched',
                            'debugging_capability': 'combination_resilience'
                        }

                        if problem_idx not in combination_variations_map:
                            combination_variations_map[problem_idx] = []
                        combination_variations_map[problem_idx].append(variation)
                        successful_variations += 1

            self.logger.debug(f"          ✅ Final variations batch: {successful_variations}/{total_requests} successful")

        except Exception as e:
            self.logger.debug(f"          ❌ Final variations batch failed: {e}")

        return combination_variations_map

    def _generate_all_variations_mega_batch(self, problem_data: List[Dict],
                                          all_problem_candidates: List[tuple],
                                          engine) -> Dict:
        """MEGA BATCH: Generate ALL variation types (generic, candidate, combination) in ONE massive parallel batch."""

        self.logger.debug(f"      🚀 Preparing MEGA BATCH with ALL variation types...")

        # Collect ALL requests across ALL types
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track (problem_idx, variation_type, data) for each request

        # 1. GENERIC TRANSFORMATIONS
        transformation_types = {
            'counterfactual': {
                'prompt': "Create a counterfactual version by changing the scenario context while keeping the exact same numerical values and mathematical relationships to preserve the identical answer.",
                'capability': 'context_preservation'
            },
            'interrogative_expansion': {
                'prompt': "Expand into a multi-part question format while maintaining the same core calculation.",
                'capability': 'format_dependency'
            },
            'rephrasing': {
                'prompt': "Rephrase using entirely different vocabulary while keeping the exact same meaning and answer.",
                'capability': 'linguistic_flexibility'
            }
        }

        for problem_idx, problem_info in enumerate(problem_data):
            problem_text = problem_info['problem_text']

            for trans_type, config in transformation_types.items():
                system_prompt = f"""You are an expert at creating question variations that test specific cognitive capabilities.

TASK: Create a {trans_type} variation of the given problem.

TRANSFORMATION GOAL: {config['prompt']}

CRITICAL REQUIREMENTS:
1. PRESERVE the exact numerical answer - the final result must be IDENTICAL
2. MAINTAIN all mathematical relationships and calculation methods
3. Use PLAIN TEXT only - no markdown formatting
4. Ensure the variation tests {config['capability']} capability
5. Make meaningful changes while keeping the same numerical answer

OUTPUT FORMAT: Return ONLY the transformed question wrapped in <question> tags.
Example: <question>Your transformed question here</question>"""

                user_prompt = f"""Original Problem: {problem_text}

Create a {trans_type} variation that {config['prompt']}

Generate ONE high-quality {trans_type} variation."""

                all_system_prompts.append(system_prompt)
                all_user_prompts.append(user_prompt)
                request_mapping.append((problem_idx, 'generic', {
                    'trans_type': trans_type,
                    'config': config
                }))

        # 2. CANDIDATE VARIATIONS
        variants_per_candidate = 4
        for problem_idx, candidate in all_problem_candidates:
            problem_info = problem_data[problem_idx]
            problem_text = problem_info['problem_text']

            highlighted_question = problem_text.replace(candidate['text'], f"→→→{candidate['text']}←←←")

            system_prompt = f"""You are an expert at generating contextually-appropriate alternative representations that will be substituted in place of specific elements in questions.

CRITICAL TASK: Generate {variants_per_candidate} different alternatives for the marked element that will be substituted IN-PLACE in the original question.

🚨 SUBSTITUTION TEST: Each variant must make sense when replacing the original element in context.

ESSENTIAL REQUIREMENTS:
1. ✅ PRESERVE exact meaning, numerical value, and format appropriateness
2. ✅ ENSURE the substituted question remains grammatically correct and coherent
3. ✅ MAINTAIN the same answer and logical structure
4. ✅ RESPECT compositional units (don't break apart formats like "H:M:S", "X hours Y minutes")
5. ✅ Use appropriate register for the question context

DIVERSITY & COMPLEXITY GUIDELINES:
🎯 CREATE meaningful, diverse alternatives (not trivial changes)
🎯 Each variant should be SUBSTANTIALLY DIFFERENT from others in:
   - Word choice and phrasing style
   - Level of formality (formal/informal/technical)
   - Complexity level (simple/detailed/descriptive)
   - Representation format (numeric/textual/abbreviated)

OUTPUT FORMAT: Return exactly {variants_per_candidate} variants, one per line:
<variants>
variant 1
variant 2
variant 3
variant 4
</variants>"""

            user_prompt = f"""ORIGINAL QUESTION: {problem_text}
HIGHLIGHTED CONTEXT: {highlighted_question}
ELEMENT TO VARY: "{candidate['text']}"
ELEMENT TYPE: {candidate['domain']} - {candidate['topic']}

🎯 TASK: Generate {variants_per_candidate} alternatives that can be substituted IN-PLACE without changing:
- The question's core meaning
- The mathematical/logical structure
- The expected answer type
- Grammatical correctness

Generate {variants_per_candidate} high-quality alternatives optimized for in-place substitution."""

            all_system_prompts.append(system_prompt)
            all_user_prompts.append(user_prompt)
            request_mapping.append((problem_idx, 'candidate', {
                'candidate': candidate
            }))

        # 3. COMBINATION VARIATIONS
        # Group candidates by problem for combination generation
        candidates_by_problem = {}
        for problem_idx, candidate in all_problem_candidates:
            if problem_idx not in candidates_by_problem:
                candidates_by_problem[problem_idx] = []
            candidates_by_problem[problem_idx].append(candidate)

        from itertools import combinations
        for problem_idx, candidates in candidates_by_problem.items():
            if len(candidates) >= 2:  # Need at least 2 candidates for combinations
                problem_text = problem_data[problem_idx]['problem_text']
                max_combination_size = min(len(candidates), 3)

                for size in range(2, max_combination_size + 1):
                    for combo in list(combinations(candidates, size))[:2]:  # Max 2 combos per size
                        # Create transformation details
                        transformation_details = []
                        for candidate in combo:
                            fallback_variant = self._get_simple_variant(candidate)
                            transformation_details.append(f"'{candidate['text']}' → '{fallback_variant}'")

                        system_prompt = f"""You are an expert at creating meaningful question variations by transforming multiple candidates simultaneously.

CRITICAL REQUIREMENTS:
1. PRESERVE exact numerical answer and question intent
2. USE EXACTLY the specified transformations provided
3. TRANSFORM all specified candidates using the given transformations
4. ENSURE mathematical/logical validity
5. Use PLAIN TEXT only - no formatting

OUTPUT FORMAT: Return ONLY the modified question wrapped in <question> tags.
Example: <question>Your modified question here</question>"""

                        user_prompt = f"""Original: {problem_text}

Apply these EXACT transformations:
{chr(10).join(transformation_details)}

Create ONE variation using EXACTLY the specified transformations while preserving exact answer and intent."""

                        all_system_prompts.append(system_prompt)
                        all_user_prompts.append(user_prompt)
                        request_mapping.append((problem_idx, 'combination', {
                            'candidates': combo,
                            'size': size,
                            'transformation_details': transformation_details
                        }))

        # EXECUTE MEGA BATCH
        self.logger.debug(f"      🎆 MEGA BATCH: {len(all_system_prompts)} total requests across ALL variation types!")
        self.logger.debug(f"         - Generic transformations: {sum(1 for r in request_mapping if r[1] == 'generic')}")
        self.logger.debug(f"         - Candidate variations: {sum(1 for r in request_mapping if r[1] == 'candidate')}")
        self.logger.debug(f"         - Combination variations: {sum(1 for r in request_mapping if r[1] == 'combination')}")

        all_variations_map = {}

        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 20 == 0:
                        self.logger.debug(f"        Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process ALL responses
            successful_requests = 0
            for i, (response, (problem_idx, variation_type, data)) in enumerate(zip(responses, request_mapping)):
                # Safety check for problem_idx bounds
                if problem_idx >= len(problem_data):
                    self.logger.debug(f"        ⚠️ Invalid problem_idx {problem_idx} >= {len(problem_data)}, skipping")
                    continue

                variations = self._process_mega_batch_response(response, variation_type, data, problem_data[problem_idx])

                if variations:
                    if problem_idx not in all_variations_map:
                        all_variations_map[problem_idx] = []
                    all_variations_map[problem_idx].extend(variations)
                    successful_requests += 1

            self.logger.debug(f"      🎆 MEGA BATCH COMPLETE: {successful_requests}/{len(request_mapping)} successful requests")

        except Exception as e:
            self.logger.debug(f"      ❌ MEGA BATCH failed: {e}")

        return all_variations_map

    def _process_mega_batch_response(self, response: str, variation_type: str, data: Dict, problem_info: Dict) -> List[Dict]:
        """Process a single response from the mega batch based on its type."""
        import re
        from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question

        variations = []

        if variation_type == 'generic':
            question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
            if question_match:
                cleaned = clean_model_response(question_match.group(1).strip())
                if cleaned and is_valid_question(cleaned):
                    trans_type = data['trans_type']
                    config = data['config']
                    variation = {
                        'original_problem': problem_info['problem_text'],
                        'modified_problem': cleaned,
                        'transformation_type': f'mega_batch_generic_{trans_type}',
                        'generation_method': 'mega_batch_generic',
                        'detection_method': 'pattern_based',
                        'debugging_capability': config['capability'],
                        'confidence': 'mega_batch_generated',
                        'domains_involved': ['generic'],
                        'original_component': trans_type,
                        'new_component': f'{trans_type}_variation',
                        'combination_size': 0,
                        'cross_domain': False
                    }
                    variations.append(variation)

        elif variation_type == 'candidate':
            variants_match = re.search(r'<variants>(.*?)</variants>', response, re.DOTALL)
            if variants_match:
                variant_lines = variants_match.group(1).strip().split('\n')
                raw_variants = [line.strip() for line in variant_lines if line.strip()]

                candidate = data['candidate']
                for j, variant in enumerate(raw_variants[:4]):  # Max 4 variants
                    if variant and variant != candidate['text']:
                        variation = {
                            'original_problem': problem_info['problem_text'],
                            'modified_problem': problem_info['problem_text'].replace(candidate['text'], variant),
                            'transformation_type': f'mega_batch_candidate_{candidate["domain"]}',
                            'candidates_transformed': [candidate['text']],
                            'selected_variants': {candidate['text']: variant},
                            'transformation_details': [f"'{candidate['text']}' → '{variant}'"],
                            'combination_size': 1,
                            'domains_involved': [candidate['domain']],
                            'cross_domain': False,
                            'confidence': 'mega_batch_generated',
                            'debugging_capability': candidate.get('topic', 'unknown'),
                            'generation_method': 'mega_batch_candidate',
                            'detection_method': 'pattern_based',
                            'original_component': candidate['text'],
                            'new_component': variant
                        }
                        variations.append(variation)

        elif variation_type == 'combination':
            question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
            if question_match:
                cleaned = clean_model_response(question_match.group(1).strip())
                if cleaned and is_valid_question(cleaned):
                    combo_data = data
                    variation = {
                        'original_problem': problem_info['problem_text'],
                        'modified_problem': cleaned,
                        'transformation_type': f'mega_batch_combination_{combo_data["size"]}way',
                        'candidates_transformed': [c['text'] for c in combo_data['candidates']],
                        'transformation_details': combo_data['transformation_details'],
                        'combination_size': combo_data['size'],
                        'domains_involved': list(set(c['domain'] for c in combo_data['candidates'])),
                        'cross_domain': len(set(c['domain'] for c in combo_data['candidates'])) > 1,
                        'confidence': 'mega_batch_generated',
                        'generation_method': 'mega_batch_combination',
                        'detection_method': 'pattern_based'
                    }
                    variations.append(variation)

        return variations

    def _generate_cross_problem_candidate_variations(self, problem_data: List[Dict],
                                                   all_problem_candidates: List[tuple],
                                                   engine) -> Dict:
        """Generate variations for all candidates across all problems in batched API calls."""

        if not all_problem_candidates:
            return {}

        # Prepare ALL candidate variation requests across ALL problems
        system_prompts = []
        user_prompts = []
        request_mapping = []  # Track (problem_idx, candidate) for each request
        variants_per_candidate = 4

        self.logger.debug(f"      🚀 Preparing {len(all_problem_candidates)} candidate variation requests...")

        for problem_idx, candidate in all_problem_candidates:
            problem_info = problem_data[problem_idx]
            problem_text = problem_info['problem_text']

            # Generate context-aware prompt (same as original but with problem mapping)
            highlighted_question = problem_text.replace(candidate['text'], f"→→→{candidate['text']}←←←")

            system_prompt = f"""You are an expert at generating contextually-appropriate alternative representations that will be substituted in place of specific elements in questions.

CRITICAL TASK: Generate {variants_per_candidate} different alternatives for the marked element that will be substituted IN-PLACE in the original question.

🚨 SUBSTITUTION TEST: Each variant must make sense when replacing the original element in context.

ESSENTIAL REQUIREMENTS:
1. ✅ PRESERVE exact meaning, numerical value, and format appropriateness
2. ✅ ENSURE the substituted question remains grammatically correct and coherent
3. ✅ MAINTAIN the same answer and logical structure
4. ✅ RESPECT compositional units (don't break apart formats like "H:M:S", "X hours Y minutes")
5. ✅ Use appropriate register for the question context

DIVERSITY & COMPLEXITY GUIDELINES:
🎯 CREATE meaningful, diverse alternatives (not trivial changes)
🎯 Each variant should be SUBSTANTIALLY DIFFERENT from others in:
   - Word choice and phrasing style
   - Level of formality (formal/informal/technical)
   - Complexity level (simple/detailed/descriptive)
   - Representation format (numeric/textual/abbreviated)

OUTPUT FORMAT: Return exactly {variants_per_candidate} variants, one per line:
<variants>
variant 1
variant 2
variant 3
variant 4
</variants>"""

            user_prompt = f"""ORIGINAL QUESTION: {problem_text}
HIGHLIGHTED CONTEXT: {highlighted_question}
ELEMENT TO VARY: "{candidate['text']}"
ELEMENT TYPE: {candidate['domain']} - {candidate['topic']}

🎯 TASK: Generate {variants_per_candidate} alternatives that can be substituted IN-PLACE without changing:
- The question's core meaning
- The mathematical/logical structure
- The expected answer type
- Grammatical correctness

Generate {variants_per_candidate} high-quality alternatives optimized for in-place substitution."""

            system_prompts.append(system_prompt)
            user_prompts.append(user_prompt)
            request_mapping.append((problem_idx, candidate))

        # MASSIVE BATCH CALL: Process ALL candidate requests across ALL problems at once
        self.logger.debug(f"      📦 Executing massive batch call for {len(system_prompts)} requests...")

        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(system_prompts, user_prompts)
            else:
                # Fallback for non-batching clients
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(system_prompts, user_prompts)):
                    if i % 10 == 0:  # Progress indicator
                        self.logger.debug(f"        Progress: {i}/{len(system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process all responses and map back to problems
            candidate_variations_map = {}
            successful_requests = 0

            for i, (response, (problem_idx, candidate)) in enumerate(zip(responses, request_mapping)):
                # Extract and validate variants (same logic as original)
                import re
                variants_match = re.search(r'<variants>(.*?)</variants>', response, re.DOTALL)
                if variants_match:
                    variant_lines = variants_match.group(1).strip().split('\n')
                    raw_variants = [line.strip() for line in variant_lines if line.strip()]

                    # Create variation objects
                    variations = []
                    for j, variant in enumerate(raw_variants[:4]):  # Max 4 variants
                        if variant and variant != candidate['text']:
                            variation = {
                                'original_problem': problem_data[problem_idx]['problem_text'],
                                'modified_problem': problem_data[problem_idx]['problem_text'].replace(candidate['text'], variant),
                                'transformation_type': f'cross_batch_candidate_{candidate["domain"]}',
                                'candidates_transformed': [candidate['text']],
                                'selected_variants': {candidate['text']: variant},
                                'transformation_details': [f"'{candidate['text']}' → '{variant}'"],
                                'combination_size': 1,
                                'domains_involved': [candidate['domain']],
                                'cross_domain': False,
                                'confidence': 'cross_batch_generated',
                                'debugging_capability': candidate.get('topic', 'unknown'),
                                'generation_method': 'cross_problem_batched',
                                'detection_method': 'pattern_based',
                                'original_component': candidate['text'],
                                'new_component': variant
                            }
                            variations.append(variation)

                    if variations:
                        # Use a string key instead of dict key to avoid unhashable type error
                        key = f"problem_{problem_idx}_candidate_{candidate['text']}"
                        candidate_variations_map[key] = {
                            'problem_idx': problem_idx,
                            'candidate': candidate,
                            'variations': variations
                        }
                        successful_requests += 1

            self.logger.debug(f"      ✅ Batch complete: {successful_requests}/{len(request_mapping)} successful candidate variations")

        except Exception as e:
            self.logger.debug(f"      ❌ Cross-problem batch failed: {e}")
            candidate_variations_map = {}

        return candidate_variations_map

    def _generate_cross_problem_generic_transformations(self, problem_data: List[Dict], engine) -> Dict:
        """Generate generic transformations for ALL problems in batched API calls."""
        self.logger.debug(f"      🚀 Preparing generic transformation requests for {len(problem_data)} problems...")

        # Collect all generic transformation requests across ALL problems
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track (problem_idx, transformation_type) for each request

        # Transformation types from the original engine
        transformation_types = {
            'counterfactual': {
                'prompt': "Create a counterfactual version by changing the scenario context while keeping the exact same numerical values and mathematical relationships to preserve the identical answer.",
                'capability': 'context_preservation'
            },
            'interrogative_expansion': {
                'prompt': "Expand into a multi-part question format while maintaining the same core calculation.",
                'capability': 'format_dependency'
            },
            'logical_formulation': {
                'prompt': "Reformulate using logical/mathematical language while preserving the exact same problem.",
                'capability': 'mathematical_equivalence'
            },
            'rephrasing': {
                'prompt': "Rephrase using entirely different vocabulary while keeping the exact same meaning and answer.",
                'capability': 'linguistic_flexibility'
            },
            'domain_shift': {
                'prompt': "Shift the context/domain while preserving the exact mathematical relationship and answer.",
                'capability': 'context_preservation'
            }
        }

        for problem_idx, problem_info in enumerate(problem_data):
            problem_text = problem_info['problem_text']

            for trans_type, config in transformation_types.items():
                system_prompt = f"""You are an expert at creating question variations that test specific cognitive capabilities.

TASK: Create a {trans_type} variation of the given problem.

TRANSFORMATION GOAL: {config['prompt']}

CRITICAL REQUIREMENTS:
1. PRESERVE the exact numerical answer - the final result must be IDENTICAL
2. MAINTAIN all mathematical relationships and calculation methods
3. Use PLAIN TEXT only - no markdown formatting
4. Ensure the variation tests {config['capability']} capability
5. Make meaningful changes while keeping the same numerical answer

OUTPUT FORMAT: Return ONLY the transformed question wrapped in <question> tags.
Example: <question>Your transformed question here</question>"""

                user_prompt = f"""Original Problem: {problem_text}

Create a {trans_type} variation that {config['prompt']}

Generate ONE high-quality {trans_type} variation."""

                all_system_prompts.append(system_prompt)
                all_user_prompts.append(user_prompt)
                request_mapping.append((problem_idx, trans_type, config))

        # MASSIVE BATCH CALL for all generic transformations
        self.logger.debug(f"      📦 Executing massive batch call for {len(all_system_prompts)} generic transformation requests...")

        generic_variations_map = {}
        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 5 == 0:
                        self.logger.debug(f"        Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses and group by problem
            successful_transformations = 0
            for i, (response, (problem_idx, trans_type, config)) in enumerate(zip(responses, request_mapping)):
                import re
                question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
                if question_match:
                    from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question
                    cleaned = clean_model_response(question_match.group(1).strip())

                    if cleaned and is_valid_question(cleaned):
                        variation = {
                            'original_problem': problem_data[problem_idx]['problem_text'],
                            'modified_problem': cleaned,
                            'transformation_type': f'generic_{trans_type}',
                            'generation_method': 'cross_problem_generic_batched',
                            'detection_method': 'pattern_based',
                            'debugging_capability': config['capability'],
                            'confidence': 'cross_batch_generated',
                            'domains_involved': ['generic'],
                            'original_component': trans_type,
                            'new_component': f'{trans_type}_variation',
                            'combination_size': 0,
                            'cross_domain': False
                        }

                        if problem_idx not in generic_variations_map:
                            generic_variations_map[problem_idx] = []
                        generic_variations_map[problem_idx].append(variation)
                        successful_transformations += 1

            self.logger.debug(f"      ✅ Generic batch complete: {successful_transformations}/{len(request_mapping)} successful transformations")

        except Exception as e:
            self.logger.debug(f"      ❌ Cross-problem generic batch failed: {e}")

        return generic_variations_map

    def _generate_cross_problem_combination_variations(self, problem_data: List[Dict],
                                                     all_problem_candidates: List[tuple],
                                                     engine) -> Dict:
        """Generate combination variations for ALL problems using cross-problem batching."""
        self.logger.debug(f"      🚀 Preparing combination variation requests...")

        combination_variations_map = {}

        # Group candidates by problem for combination generation
        candidates_by_problem = {}
        for problem_idx, candidate in all_problem_candidates:
            if problem_idx not in candidates_by_problem:
                candidates_by_problem[problem_idx] = []
            candidates_by_problem[problem_idx].append(candidate)

        # Generate combinations for each problem and collect ALL requests
        all_system_prompts = []
        all_user_prompts = []
        request_mapping = []  # Track (problem_idx, combination_data) for each request

        for problem_idx, candidates in candidates_by_problem.items():
            if len(candidates) >= 2:  # Need at least 2 candidates for combinations
                problem_text = problem_data[problem_idx]['problem_text']

                # Generate combinations (2-way to 4-way)
                from itertools import combinations
                max_combination_size = min(len(candidates), 4)

                for size in range(2, max_combination_size + 1):
                    for combo in list(combinations(candidates, size))[:3]:  # Max 3 combos per size
                        # Create transformation details
                        transformation_details = []
                        for candidate in combo:
                            # Use simple fallback transformations for combinations
                            fallback_variant = self._get_simple_variant(candidate)
                            transformation_details.append(f"'{candidate['text']}' → '{fallback_variant}'")

                        system_prompt = f"""You are an expert at creating meaningful question variations by transforming multiple candidates simultaneously.

CRITICAL REQUIREMENTS:
1. PRESERVE exact numerical answer and question intent
2. USE EXACTLY the specified transformations provided
3. TRANSFORM all specified candidates using the given transformations
4. ENSURE mathematical/logical validity
5. Use PLAIN TEXT only - no formatting

OUTPUT FORMAT: Return ONLY the modified question wrapped in <question> tags.
Example: <question>Your modified question here</question>"""

                        user_prompt = f"""Original: {problem_text}

Apply these EXACT transformations:
{chr(10).join(transformation_details)}

Create ONE variation using EXACTLY the specified transformations while preserving exact answer and intent."""

                        all_system_prompts.append(system_prompt)
                        all_user_prompts.append(user_prompt)
                        request_mapping.append((problem_idx, {
                            'candidates': combo,
                            'size': size,
                            'transformation_details': transformation_details
                        }))

        if not all_system_prompts:
            self.logger.debug(f"      ℹ️ No combination requests generated")
            return combination_variations_map

        # MASSIVE BATCH CALL for all combination variations
        self.logger.debug(f"      📦 Executing massive batch call for {len(all_system_prompts)} combination requests...")

        try:
            if hasattr(engine.model_client, 'get_model_response'):
                responses = engine.model_client.get_model_response(all_system_prompts, all_user_prompts)
            else:
                responses = []
                for i, (sys_prompt, user_prompt) in enumerate(zip(all_system_prompts, all_user_prompts)):
                    if i % 5 == 0:
                        self.logger.debug(f"        Progress: {i}/{len(all_system_prompts)}")
                    response = str(engine.model_client.generate(user_prompt, sys_prompt))
                    responses.append(response)

            # Process responses
            successful_combinations = 0
            for i, (response, (problem_idx, combo_data)) in enumerate(zip(responses, request_mapping)):
                import re
                question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
                if question_match:
                    from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response, is_valid_question
                    cleaned = clean_model_response(question_match.group(1).strip())

                    if cleaned and is_valid_question(cleaned):
                        variation = {
                            'original_problem': problem_data[problem_idx]['problem_text'],
                            'modified_problem': cleaned,
                            'transformation_type': f'cross_batch_combination_{combo_data["size"]}way',
                            'candidates_transformed': [c['text'] for c in combo_data['candidates']],
                            'transformation_details': combo_data['transformation_details'],
                            'combination_size': combo_data['size'],
                            'domains_involved': list(set(c['domain'] for c in combo_data['candidates'])),
                            'cross_domain': len(set(c['domain'] for c in combo_data['candidates'])) > 1,
                            'confidence': 'cross_batch_generated',
                            'generation_method': 'cross_problem_combination_batched',
                            'detection_method': 'pattern_based'
                        }

                        if problem_idx not in combination_variations_map:
                            combination_variations_map[problem_idx] = []
                        combination_variations_map[problem_idx].append(variation)
                        successful_combinations += 1

            self.logger.debug(f"      ✅ Combination batch complete: {successful_combinations}/{len(request_mapping)} successful combinations")

        except Exception as e:
            self.logger.debug(f"      ❌ Cross-problem combination batch failed: {e}")

        return combination_variations_map

    def _get_simple_variant(self, candidate: Dict) -> str:
        """Generate a simple variant for a candidate (fallback for combinations)."""
        text = candidate['text']
        domain = candidate['domain']

        # Simple fallback transformations
        if domain == 'math' and text.isdigit():
            return f"the number {text}"
        elif domain == 'temporal' and 'minute' in text:
            return text.replace('minute', 'min')
        elif domain == 'temporal' and 'hour' in text:
            return text.replace('hour', 'hr')
        else:
            return f"the {text}"

    def _process_variation_batch_parallel(self, batch_problems: List[Dict], start_idx: int, max_workers: int):
        """Process problems in parallel using multiprocessing."""
        import concurrent.futures
        import threading

        # Thread-safe data collection
        results_lock = threading.Lock()

        def process_single_problem(problem_with_idx):
            """Process a single problem and return the results."""
            i, problem = problem_with_idx
            actual_idx = start_idx + i

            try:
                # Initialize engine for this worker (each worker gets its own engine)
                # Always create model client for variations stage (needed for combination-based variations)
                model_client = self._get_model_client_for_stage('variations')
                # Use variation model for validation (not judge model to avoid dual client initialization)

                from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
                engine = UnifiedVariationEngine(model_client=model_client, judge_model_client=None)

                self.logger.debug(f"    🔄 Worker processing problem {actual_idx + 1}")

                # Extract problem data
                if isinstance(problem, dict):
                    problem_text = problem.get('problem', problem.get('question', str(problem)))
                    problem_id = problem.get('id', f'problem_{actual_idx}')
                    ground_truth = problem.get('answer', problem.get('ground_truth', ''))
                else:
                    problem_text = str(problem)
                    problem_id = f'problem_{actual_idx}'
                    ground_truth = ''

                # SEMANTIC VERSION: Fallback not supported - use batch processing
                self.logger.debug(f"    ⚠️ Fallback to old engine method not supported in semantic version")
                self.logger.debug(f"    Returning empty variations for problem {actual_idx + 1}")
                all_variations = []

                self.logger.debug(f"    ✅ Worker completed problem {actual_idx + 1} ({len(all_variations)} variations)")
                return (problem_id, problem_text, ground_truth, all_variations, actual_idx)

            except Exception as e:
                self.logger.debug(f"    ❌ Worker failed on problem {actual_idx + 1}: {e}")
                return None

        # Process problems in parallel
        problem_indices = list(enumerate(batch_problems))
        successful_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all problems
            future_to_problem = {
                executor.submit(process_single_problem, problem_with_idx): problem_with_idx[0]
                for problem_with_idx in problem_indices
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_problem):
                problem_idx = future_to_problem[future]
                try:
                    result = future.result()
                    if result:
                        successful_results.append(result)
                except Exception as e:
                    actual_idx = start_idx + problem_idx
                    self.logger.debug(f"    ❌ Exception in worker for problem {actual_idx + 1}: {e}")

        # Process all successful results and add to self.data (thread-safe)
        self.logger.debug(f"  📊 Parallel processing complete: {len(successful_results)}/{len(batch_problems)} problems successful")
        with results_lock:
            for problem_id, problem_text, ground_truth, all_variations, actual_idx in successful_results:
                self._create_problem_entries(problem_id, problem_text, ground_truth, all_variations)

    def _create_problem_entries(self, problem_id: str, problem_text: str, ground_truth: str, variations: List[Dict]):
        """Create baseline and variant entries for a problem."""
        timestamp = datetime.now().isoformat()

        # Create baseline entry
        baseline_entry = {
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

            # Response placeholders
            'has_model_response': False,
            'model_response': '',
            'model_thinking': '',
            'model_final_answer': '',
            'response_generation_time': 0.0,
            'response_success': False,
            'response_error_message': '',
            'response_timestamp': 0,

            # Evaluation placeholders
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
        self.data.append(baseline_entry)

        # Create variant entries
        for j, variation in enumerate(variations):
            variant_entry = {
                'problem_id': problem_id,
                'variation_id': f'{problem_id}_variant_{j+1}',
                'variation_index': j + 1,
                'variation_type': 'variant',
                'is_baseline': False,
                'is_variant': True,
                'variant_number': j + 1,
                'total_variants_for_problem': len(variations),

                # Problem content
                'original_problem': problem_text,
                'modified_problem': variation.get('modified_problem', ''),
                'baseline_problem': problem_text,
                'ground_truth_answer': ground_truth,

                # Transformation details from variation
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

                # Stage tracking
                'stages_completed': ['variations'],
                'variation_generation_timestamp': timestamp,

                # Response placeholders
                'has_model_response': False,
                'model_response': '',
                'model_thinking': '',
                'model_final_answer': '',
                'response_generation_time': 0.0,
                'response_success': False,
                'response_error_message': '',
                'response_timestamp': 0,

                # Evaluation placeholders
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
            self.data.append(variant_entry)

    def _run_cagrad_batched(self, problem: str, answer: str, clusters, embedder=None):
        """
        Rank clusters by CAGrad gradient and return top-k most important clusters.

        NEW APPROACH: Instead of pairwise dependency testing, simply rank clusters
        by their individual gradient scores and return the top-k most impactful ones.

        Args:
            problem: Original problem text
            answer: Expected answer
            clusters: List of SemanticCluster objects
            embedder: SentenceTransformer for principled counterfactual replacement (optional)

        Returns:
            List of (cluster_id, gradient_score) tuples, ranked by importance
        """
        import math

        max_clusters = self.config.get('max_cagrad_clusters', 10)

        self.logger.debug(f"      🧪 CAGrad: Ranking {len(clusters)} clusters by gradient score...")
        self.logger.debug(f"         Will select top-{max_clusters} clusters for variation generation")
        if embedder:
            self.logger.debug(f"         Using embedding-based counterfactuals (principled approach)")
        else:
            self.logger.debug(f"         Using [MASK] token (fallback)")

        # Step 1: Collect prompts for baseline + individual clusters
        all_prompts = []
        prompt_types = []

        # Baseline
        all_prompts.append(problem)
        prompt_types.append({'type': 'baseline'})

        # Individual cluster masks
        for i, cluster in enumerate(clusters):
            masked = self._mask_cluster_spans(problem, cluster.spans, embedder)
            all_prompts.append(masked)
            prompt_types.append({'type': 'individual', 'cluster_id': i})

        # Step 2: Get model client (reuse from config)
        model_client = self._get_model_client_for_stage('variations')

        # Step 3: Batch inference for ALL prompts at once (NO MANUAL CHUNKING!)
        self.logger.debug(f"         Batching {len(all_prompts)} prompts for CAGrad testing...")
        system_prompts = ["You are a helpful assistant. Provide concise, direct answers."] * len(all_prompts)

        results = model_client.get_model_response_with_logprobs(
            system_prompts,
            all_prompts,
            max_new_tokens=50,
            temperature=0.1
        )

        # Step 4: Extract probabilities
        probabilities = []
        for result in results:
            logprobs = result.get('logprobs', [])
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                prob = min(math.exp(avg_logprob), 1.0)
            else:
                prob = 0.5
            probabilities.append(prob)

        # Step 5: Calculate individual gradients and rank
        baseline_prob = probabilities[0]
        cluster_rankings = []

        for i in range(len(clusters)):
            masked_prob = probabilities[i + 1]
            gradient = abs(baseline_prob - masked_prob)
            cluster_rankings.append((i, gradient))

        # Sort by gradient (highest to lowest)
        cluster_rankings.sort(key=lambda x: x[1], reverse=True)

        # Take top-k clusters
        top_k_clusters = cluster_rankings[:max_clusters]

        # Print rankings
        self.logger.debug(f"\n         📊 Cluster Rankings (by CAGrad gradient):")
        self.logger.debug(f"         {'='*60}")
        for rank, (cluster_id, gradient) in enumerate(top_k_clusters, 1):
            cluster_texts = clusters[cluster_id].texts
            cluster_preview = ", ".join(cluster_texts[:3])
            if len(cluster_texts) > 3:
                cluster_preview += f" (+{len(cluster_texts)-3} more)"
            self.logger.debug(f"         #{rank:2d}. Cluster {cluster_id}: [{cluster_preview}]")
            self.logger.debug(f"              Gradient: {gradient:.4f}")

        self.logger.debug(f"\n         ✅ Selected top-{len(top_k_clusters)} clusters for variation generation")
        return top_k_clusters

    def _mask_cluster_spans(self, text: str, spans, embedder=None):
        """
        Replace spans with semantically opposite words using embedding distance.

        Uses the same embedder as clustering for principled counterfactuals.
        Falls back to [MASK] if no embedder provided.

        Args:
            text: Original text
            spans: List of (start, end) positions to replace
            embedder: SentenceTransformer for finding semantic opposites (optional)

        Returns:
            Text with spans replaced by semantically opposite words or [MASK]
        """
        if not embedder:
            # Fallback to [MASK]
            masked = text
            for span in sorted(spans, reverse=True):
                masked = masked[:span[0]] + "[MASK]" + masked[span[1]:]
            return masked

        # Use embedding-based replacement
        from scipy.spatial.distance import cosine
        import numpy as np
        import re

        # Type-specific vocabularies (same as in semantic_composite_detector.py)
        vocabularies = {
            'number': ['1', '2', '3', '5', '7', '10', '15', '20', '25', '50', '100'],
            'entity': ['Alex', 'Jordan', 'Taylor', 'Sam', 'Morgan', 'Casey', 'Riley'],
            'object': ['apple', 'book', 'car', 'pen', 'table', 'chair', 'box'],
            'temporal': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'default': ['something', 'anything', 'nothing', 'everything', 'item', 'thing', 'object']
        }

        def infer_type(t):
            t = t.strip()
            if re.match(r'^-?\d+\.?\d*$', t):
                return 'number'
            if t and t[0].isupper() and len(t) > 1 and t.isalpha():
                return 'entity'
            if t.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                return 'temporal'
            return 'default'

        def find_opposite(original_text):
            # Infer type and get vocabulary
            cand_type = infer_type(original_text)
            vocab = vocabularies.get(cand_type, vocabularies['default'])

            # Get embeddings
            original_emb = embedder.encode([original_text], show_progress_bar=False)[0]
            vocab_embs = embedder.encode(vocab, show_progress_bar=False)

            # Find most distant word
            distances = [cosine(original_emb, ve) for ve in vocab_embs]
            valid_indices = [i for i, word in enumerate(vocab) if word.lower() != original_text.lower()]

            if valid_indices:
                max_idx = max(valid_indices, key=lambda i: distances[i])
                return vocab[max_idx]
            else:
                return vocab[np.argmax(distances)]

        # Replace each span
        masked = text
        for span in sorted(spans, reverse=True):
            original_text = text[span[0]:span[1]]
            replacement = find_opposite(original_text)
            masked = masked[:span[0]] + replacement + masked[span[1]:]

        return masked


def main():
    """Main function with command line interface for batched processing."""

    parser = argparse.ArgumentParser(
        description="Unified Batched Pipeline with Progressive Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages with batching
  python unified_batched_pipeline.py --input large_problems.jsonl --unified-file results.json --all-stages --batch-size 100

  # Run specific stages with batching
  python unified_batched_pipeline.py --unified-file results.json --stage variations --input problems.jsonl --batch-size 50
  python unified_batched_pipeline.py --unified-file results.json --stage responses --batch-size 30

  # Large dataset processing - automatically optimized batching
  python unified_batched_pipeline.py --input 10k_problems.jsonl --unified-file large_results.json --all-stages --batch-size 200 --max-workers 8

  # Performance tuning examples:
  # Small dataset (few problems)
  python unified_batched_pipeline.py --input small_problems.jsonl --unified-file results.json --all-stages --batch-size 10 --max-workers 2

  # Large dataset (many problems) - maximize parallelization
  python unified_batched_pipeline.py --input large_problems.jsonl --unified-file results.json --all-stages --batch-size 100 --max-workers 8

  # High variation complexity
  python unified_batched_pipeline.py --input complex_problems.jsonl --unified-file results.json --all-stages --max-combinations 75
        """
    )

    # Required
    parser.add_argument('--unified-file', required=True,
                       help='Unified JSON file (created/updated progressively)')

    # Stage control
    parser.add_argument('--stage', choices=['candidate-detection', 'variations', 'validation', 'responses', 'evaluation', 'export-csv'],
                       help='Run specific stage only')
    parser.add_argument('--all-stages', action='store_true',
                       help='Run all stages in sequence')
    parser.add_argument('--stages-1-4', action='store_true',
                       help='Run stages 1-4 (variations through export, loads candidates from unified file)')

    # Input (required for variations stage)
    parser.add_argument('--input',
                       help='Input problems file (.json or .jsonl) - required for variations stage')

    # Batching configuration
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for variation generation (default: 50)')
    parser.add_argument('--response-eval-batch-size', type=int, default=None,
                       help='Batch size for response generation and evaluation (default: same as batch-size)')
    parser.add_argument('--rits-batch-size', type=int, default=None,
                       help='Batch size for RITS API calls (default: min(batch_size, 10))')
    parser.add_argument('--max-problems', type=int, default=None,
                       help='Maximum number of problems to process (default: all)')
    parser.add_argument('--save-every-batch', action='store_true', default=True,
                       help='Save after every batch (default: True)')

    # Configuration
    parser.add_argument('--client-type', default='rits', choices=['rits', 'openai','vllm'],
                       help='Model client type (default: rits)')
    parser.add_argument('--model-name', default='mistral_small_3_2_instruct',
                       help='Model for variation generation (default: mistral_small_3_2_instruct)')
    parser.add_argument('--response-model', default='mistral_small_3_2_instruct',
                       help='Model for response generation (default: mistral_small_3_2_instruct)')
    parser.add_argument('--use-llm-judge', action='store_true',
                       help='Use LLM judge for answer evaluation (default: string matching)')
    parser.add_argument('--judge-model', default=None,
                       help='Model to use as LLM judge (default: same as model-name)')
    parser.add_argument('--disable-cot', action='store_true',
                       help='Disable chain-of-thought reasoning for faster response generation')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate responses even if they already exist')
    parser.add_argument('--rectify-invalid', action='store_true',
                       help='Rectify invalid variations instead of dropping them (default: False, drops invalid)')
    parser.add_argument('--num-variations', type=int, default=3,
                       help='Number of variations per problem (default: 3)')
    parser.add_argument('--max-combinations', type=int, default=DEFAULT_MAX_COMBINATIONS,
                       help=f'Max combinations per problem for model selection (default: {DEFAULT_MAX_COMBINATIONS})')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Max workers for parallel processing (default: 4)')
    parser.add_argument('--max-model-len', type=int, default=8192,
                       help='Maximum model context length for VLLM (default: 8192)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for response generation (default: 0.1)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens to generate in response (default: 1024)')
    # parser.add_argument('--use-optimized', action='store_true',
                       # help='Use optimized LLM-guided selector')

    # Variation type configuration
    parser.add_argument('--use-generic', action='store_true', default=True, dest='use_generic',
                       help='Generate generic transformations (counterfactual, rephrasing, etc.) (default: True)')
    parser.add_argument('--no-generic', action='store_false', dest='use_generic',
                       help='Skip generic transformations - use only cluster-based variations')
    parser.add_argument('--use-cluster-variations', action='store_true', default=True, dest='use_cluster_variations',
                       help='Generate cluster-based/decomposition variations from semantic clusters (default: True)')
    parser.add_argument('--no-cluster-variations', action='store_false', dest='use_cluster_variations',
                       help='Skip cluster-based variations - use only generic/persona/long-context')
    parser.add_argument('--use-persona', action='store_true', default=False,
                       help='Generate persona-based variations (different perspectives/roles) (default: False)')
    parser.add_argument('--use-long-context', action='store_true', default=False,
                       help='Generate long-context variations for prompts >500 chars (structure/formatting/clarity) (default: False)')

    # CAGrad configuration
    parser.add_argument('--use-cagrad', action='store_true',
                       help='Use CAGrad brittleness-based variation generation instead of combination selection (default: False)')
    parser.add_argument('--cagrad-use-gradient-pruning', action='store_true',
                       help='Enable Phase 1 gradient pruning in CAGrad (default: False)')
    parser.add_argument('--cagrad-num-counterfactuals', type=int, default=8,
                       help='Number of counterfactuals to generate per fragment in CAGrad (default: 8)')
    parser.add_argument('--cagrad-pruning-threshold', type=float, default=0.3,
                       help='Pruning threshold for CAGrad Phase 1 - keep top X%% fragments (default: 0.3)')
    parser.add_argument('--cagrad-top-k', type=int, default=10,
                       help='Number of most brittle fragments to return per problem in CAGrad (default: 10)')
    parser.add_argument('--cagrad-min-fragments', type=int, default=3,
                       help='Minimum number of fragments to keep after pruning in CAGrad (default: 3)')
    parser.add_argument('--cagrad-max-fragments', type=int, default=None,
                       help='Maximum number of fragments to keep after pruning in CAGrad (default: None)')
    parser.add_argument('--filter-redundant-singles', action='store_true',
                       help='Filter single-element combinations if candidate appears in multi-element (default: False). '
                            'WARNING: May reduce drift detection as testing [A,B] together differs from testing [A] alone.')

    # Semantic clustering configuration
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='Sentence-transformers model for semantic clustering (default: all-MiniLM-L6-v2). '
                            'Recommended alternatives: all-mpnet-base-v2 (best quality), BAAI/bge-base-en-v1.5 (newer), '
                            'BAAI/bge-small-en-v1.5 (fast), all-MiniLM-L12-v2 (better than L6)')
    parser.add_argument('--semantic-threshold', type=float, default=0.35,
                       help='Distance threshold for hierarchical clustering (default: 0.35). '
                            'Lower = stricter clustering (fewer merges), Higher = more merging')
    parser.add_argument('--max-cagrad-clusters', type=int, default=10,
                       help='Maximum number of top-ranked clusters to use for variation generation when CAGrad is enabled (default: 10). '
                            'Clusters are ranked by gradient score (impact on answer), and only top-k are used for variations.')

    # Prioritized testing (speedup via logprob-based filtering)
    parser.add_argument('--use-prioritized-testing', action='store_true',
                       help='Enable prioritized testing: score variations with logprobs, test only high-priority ones (default: False)')
    parser.add_argument('--test-budget', type=float, default=1.0,
                       help='Percentage of variations to test when using prioritized testing (0.0-1.0, default: 1.0 = test all). '
                            'Example: --test-budget 0.3 tests top 30%% (3.3x speedup)')
    parser.add_argument('--sampling-method',
                       choices=['stratified', 'horvitz_thompson', 'neyman_facility'],
                       default='stratified',
                       help='Sampling method for prioritized testing:\n'
                            '  stratified: Ensures diversity (simple)\n'
                            '  horvitz_thompson: Unbiased estimates with CI (statistical rigor)\n'
                            '  neyman_facility: Maximum coverage with multi-signal scoring (OPTIMAL for finding drifts)')
    parser.add_argument('--ht-sampling-stage', choices=['before', 'after', 'both'], default='before',
                       help='When to apply HT sampling: before variation generation, after, or both')
    parser.add_argument('--ht-confidence-level', type=float, default=0.95,
                       help='Confidence level for HT estimation (0.90, 0.95, or 0.99)')
    parser.add_argument('--coverage-penalty', type=float, default=0.5,
                       help='Weight on diversity/coverage vs drift score for neyman_facility (0.0-1.0)')
    parser.add_argument('--multi-signal-scoring', action='store_true', default=True,
                       help='Use multi-signal scoring (gradient + structural + diversity) for neyman_facility')

    # Output
    parser.add_argument('--csv-output',
                       help='CSV output path (default: unified_file.csv)')

    args = parser.parse_args()

    # Validation
    if args.all_stages and not args.input:
        self.logger.debug("🧪 No --input specified, will use hardcoded test example")
        args.input = 'test'  # Set to trigger test mode

    if args.stage == 'variations' and not args.input:
        self.logger.debug("🧪 No --input specified, will use hardcoded test example")
        args.input = 'test'  # Set to trigger test mode

    # Configuration
    response_eval_batch_size = args.response_eval_batch_size or args.batch_size

    config = {
        'unified_file': args.unified_file,
        'input_problems': args.input,
        'client_type': args.client_type,
        'model_name': args.model_name,
        'response_model': args.response_model,
        'num_variations': args.num_variations,
        'max_combinations': args.max_combinations,
        'max_workers': args.max_workers,
        'max_model_len': args.max_model_len,
        'temperature': args.temperature,
        'max_new_tokens': args.max_tokens,
        'use_llm_judge': args.use_llm_judge,
        'judge_model': args.judge_model,
        'disable_cot': args.disable_cot,
        'force_regenerate': args.force_regenerate,
        'rectify_invalid': args.rectify_invalid,
        'use_model_client': True,
        'variation_config': {},
        'batch_size': args.batch_size,
        'response_eval_batch_size': response_eval_batch_size,
        'rits_batch_size': args.rits_batch_size or min(args.batch_size, 10),
        'max_problems': args.max_problems,
        'save_every_batch': args.save_every_batch,
        # Always use maximum batching - this is a batched pipeline!
        'use_batched_processing': True,
        # Variation type configuration
        'use_generic': args.use_generic,
        'use_cluster_variations': args.use_cluster_variations,
        'use_persona': args.use_persona,
        'use_long_context': args.use_long_context,
        # CAGrad configuration
        'use_cagrad': args.use_cagrad,
        'cagrad_use_gradient_pruning': args.cagrad_use_gradient_pruning,
        'cagrad_num_counterfactuals': args.cagrad_num_counterfactuals,
        'cagrad_pruning_threshold': args.cagrad_pruning_threshold,
        'cagrad_top_k': args.cagrad_top_k,
        'cagrad_min_fragments': args.cagrad_min_fragments,
        'cagrad_max_fragments': args.cagrad_max_fragments,
        'filter_redundant_singles': args.filter_redundant_singles,
        # Semantic clustering configuration
        'embedding_model': args.embedding_model,
        'semantic_threshold': args.semantic_threshold,
        'max_cagrad_clusters': args.max_cagrad_clusters,
        # Prioritized testing configuration
        'use_prioritized_testing': args.use_prioritized_testing,
        'test_budget': args.test_budget,
        'sampling_method': args.sampling_method,
        'ht_sampling_stage': args.ht_sampling_stage,
        'ht_confidence_level': args.ht_confidence_level,
        'coverage_penalty': args.coverage_penalty,
        'use_multi_signal_scoring': args.multi_signal_scoring
    }

    # Initialize batched enhancer
    enhancer = UnifiedBatchedPipeline(config)

    self.logger.debug(f"🚀 Unified Batched Pipeline Starting...")
    self.logger.debug(f"   Unified file: {args.unified_file}")
    self.logger.debug(f"   Variation batch size: {args.batch_size}")
    self.logger.debug(f"   Response/Eval batch size: {response_eval_batch_size}")
    self.logger.debug(f"   Max workers: {args.max_workers}")

    # Always use maximum batching approach
    self.logger.debug(f"   Batching strategy: 🚀 Maximum batching (API + problem-level parallelization with {config['max_workers']} workers)")

    # Variation type info
    self.logger.debug(f"   📋 Variation Types:")
    self.logger.debug(f"      - Generic: {'ENABLED' if args.use_generic else 'DISABLED'}")
    self.logger.debug(f"      - Cluster-based: {'ENABLED' if args.use_cluster_variations else 'DISABLED'} (from semantic clusters)")
    self.logger.debug(f"      - Persona: {'ENABLED' if args.use_persona else 'DISABLED'}")
    self.logger.debug(f"      - Long Context: {'ENABLED' if args.use_long_context else 'DISABLED'} (applies to prompts >500 chars)")

    # CAGrad info
    if args.use_cagrad:
        self.logger.debug(f"   🎯 CAGrad: ENABLED (brittleness-based variation generation)")
        self.logger.debug(f"      - Gradient pruning: {'ENABLED' if args.cagrad_use_gradient_pruning else 'DISABLED'}")
        self.logger.debug(f"      - Counterfactuals per fragment: {args.cagrad_num_counterfactuals}")
        self.logger.debug(f"      - Top-K fragments: {args.cagrad_top_k}")
        self.logger.debug(f"      - Pruning threshold: {args.cagrad_pruning_threshold}")
        self.logger.debug(f"      - Min fragments after pruning: {args.cagrad_min_fragments}")
        max_frags_str = str(args.cagrad_max_fragments) if args.cagrad_max_fragments else "None (unlimited)"
        self.logger.debug(f"      - Max fragments after pruning: {max_frags_str}")
    else:
        self.logger.debug(f"   🔄 CAGrad: DISABLED (using standard combination-based generation)")

    # Prioritized testing info
    if args.use_prioritized_testing:
        self.logger.debug(f"   ⚡ Prioritized Testing: ENABLED (logprob-based speedup)")
        self.logger.debug(f"      - Test budget: {int(args.test_budget*100)}%")
        self.logger.debug(f"      - Expected speedup: {1/args.test_budget:.1f}x")
        self.logger.debug(f"      - Strategy: Score all with logprobs, test top {int(args.test_budget*100)}%")
    else:
        self.logger.debug(f"   🔄 Prioritized Testing: DISABLED (testing all variations)")

    if args.input:
        self.logger.debug(f"   Input: {args.input}")

    # Start total timing
    total_start_time = time.time()
    stage_times = {}

    try:
        if args.all_stages:
            # Run all stages with batching and timing
            self.logger.debug(f"\n⏱️  Starting pipeline execution timing...")

            # Stage 1: Variations
            stage_start = time.time()
            enhancer.stage1_generate_variations_batched()
            stage_times['variations'] = time.time() - stage_start

            # Stage 2: Responses
            stage_start = time.time()
            enhancer.stage2_generate_responses()  # Now uses batched engine from updated base class
            stage_times['responses'] = time.time() - stage_start

            # Stage 3: Evaluation
            stage_start = time.time()
            enhancer.stage3_add_evaluation_metrics()  # Now uses batched engine from updated base class
            stage_times['evaluation'] = time.time() - stage_start

            # Stage 4: CSV Export
            stage_start = time.time()
            csv_output = args.csv_output or args.unified_file.replace('.json', '.csv')
            enhancer.export_to_csv(csv_output)
            stage_times['csv_export'] = time.time() - stage_start

        elif args.stage == 'candidate-detection':
            stage_start = time.time()
            enhancer.stage0_candidate_detection_only()
            stage_times['candidate_detection'] = time.time() - stage_start

        elif args.stage == 'variations':
            stage_start = time.time()
            enhancer.stage1_generate_variations_batched()
            stage_times['variations'] = time.time() - stage_start

        elif args.stage == 'validation':
            stage_start = time.time()
            enhancer.stage_validation()
            stage_times['validation'] = time.time() - stage_start

        elif args.stages_1_4:
            self.logger.debug(f"\n⏱️  Starting stages 1-4 execution (loads candidates from unified file)...")

            # Set execution mode for stages 1-4
            enhancer.config['execution_mode'] = 'stages_1_4'

            # Stage 1: Variations (loads candidates from unified file)
            stage_start = time.time()
            enhancer.stage1_generate_variations_batched()
            stage_times['variations'] = time.time() - stage_start

            # Stage 2: Responses
            stage_start = time.time()
            enhancer.stage2_generate_responses()
            stage_times['responses'] = time.time() - stage_start

            # Stage 3: Evaluation
            stage_start = time.time()
            enhancer.stage3_add_evaluation_metrics()
            stage_times['evaluation'] = time.time() - stage_start

            # Stage 4: CSV Export
            stage_start = time.time()
            csv_output = args.csv_output or args.unified_file.replace('.json', '.csv')
            enhancer.export_to_csv(csv_output)
            stage_times['csv_export'] = time.time() - stage_start

        elif args.stage == 'responses':
            stage_start = time.time()
            enhancer.stage2_generate_responses()
            stage_times['responses'] = time.time() - stage_start

        elif args.stage == 'evaluation':
            stage_start = time.time()
            enhancer.stage3_add_evaluation_metrics()
            stage_times['evaluation'] = time.time() - stage_start

        elif args.stage == 'export-csv':
            stage_start = time.time()
            enhancer.export_to_csv(args.csv_output)
            stage_times['csv_export'] = time.time() - stage_start

        else:
            self.logger.debug("❌ Please specify --stage or --all-stages")
            return 1

        # Calculate total time
        total_time = time.time() - total_start_time

        # Print timing summary
        self.logger.debug(f"\n⏱️  PIPELINE TIMING SUMMARY")
        self.logger.debug(f"{'='*50}")
        self.logger.debug(f"🎯 Batching Strategy Used:")
        self.logger.debug(f"   🚀 Automatic optimization (API + problem-level parallelization with {config['max_workers']} workers)")

        self.logger.debug(f"\n⏱️  Stage Execution Times:")
        for stage, duration in stage_times.items():
            minutes = int(duration // 60)
            seconds = duration % 60
            self.logger.debug(f"   {stage.capitalize():15} {minutes:2d}m {seconds:5.2f}s")

        self.logger.debug(f"\n⏱️  Total Pipeline Time:")
        total_minutes = int(total_time // 60)
        total_seconds = total_time % 60
        self.logger.debug(f"   {'TOTAL':15} {total_minutes:2d}m {total_seconds:5.2f}s")

        self.logger.info(f"\n🎉 Batched Pipeline complete!")
        return 0

    except Exception as e:
        self.logger.debug(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())