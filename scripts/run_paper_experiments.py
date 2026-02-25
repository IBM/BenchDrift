#!/usr/bin/env python3
"""
Automated Experiment Runner for BenchDrift Paper Replication
Runs all experiments with EXACT setup from paper using VLLM for fast batched inference.

Models: Qwen3-8B, Mistral-7B, Phi-4, Granite-3.3-8B, GPT-OSS-20B
Benchmarks: GSM8K (1,319), MMLU (1,319), ToT-Arithmetic (1,000)
Variation Types: ALL enabled (generic, cluster, persona, long-context)
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchdrift.pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline


# ==============================================================================
# EXPERIMENT CONFIGURATION (FROM PAPER)
# ==============================================================================

MODELS = {
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'phi-4': 'microsoft/phi-4',
    'granite-3.3-8b': 'ibm-granite/granite-3.3-8b-instruct',
    'gpt-oss-20b': 'openai/gpt-oss-20b'
}

BENCHMARKS = {
    'gsm8k': {
        'path': 'data/gsm8k/test.jsonl',
        'count': 1319,
        'description': 'Grade school math reasoning'
    },
    'mmlu': {
        'path': 'data/mmlu/test_1319.jsonl',
        'count': 1319,
        'description': 'Multitask language understanding'
    },
    'tot_arithmetic': {
        'path': 'data/tot/test_sampled_1000.jsonl',
        'count': 1000,
        'description': 'Test-of-time arithmetic reasoning'
    }
}

# Pipeline configuration (ALL variation types enabled - matching updated implementation)
BASE_CONFIG = {
    # Variation Types - ALL ENABLED
    'use_generic': True,
    'use_cluster_variations': True,
    'use_persona': True,
    'use_long_context': True,

    # Semantic Clustering
    'embedding_model': 'all-MiniLM-L6-v2',
    'semantic_threshold': 0.35,

    # Generation Parameters
    'batch_size': 50,  # Larger batches with VLLM
    'max_workers': 8,
    'max_model_len': 8192,
    'max_new_tokens': 1024,
    'temperature': 0.1,

    # Evaluation
    'use_llm_judge': True,
    'judge_model': 'llama_3_3_70b',  # From paper
    'rectify_invalid': True,

    # Client Configuration
    # Use 'vllm_logits' for response generation to collect logit stats
    # Use 'vllm' for evaluation (judge) - no need for logits
    'client_type': 'vllm_logits',  # Collect logits for variation/response stages
    'response_client_type': 'vllm_logits',  # Collect logits for response generation
    'eval_client_type': 'vllm',  # No logits needed for judge
    'logprobs_k': 5,  # Number of top logprobs to collect per token

    # Output
    'save_every_batch': True,
    'verbose': False,
}


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_single_experiment(model_name, model_path, benchmark_name, benchmark_info, output_dir):
    """Run single experiment: model × benchmark × all variation types."""

    experiment_id = f"{model_name}_{benchmark_name}"
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"{'='*80}")
    print(f"Model: {model_name} ({model_path})")
    print(f"Benchmark: {benchmark_name} ({benchmark_info['count']} problems)")
    print(f"Variation types: generic + cluster + persona + long-context")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    exp_dir = Path(output_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    unified_file = exp_dir / f"{experiment_id}_results.json"

    # Load benchmark data
    benchmark_path = Path(benchmark_info['path'])
    if not benchmark_path.exists():
        print(f"❌ ERROR: Benchmark file not found: {benchmark_path}")
        print(f"   Please prepare benchmark data first using scripts/prepare_benchmarks.py")
        return None

    # Configure pipeline
    config = BASE_CONFIG.copy()
    config.update({
        'unified_file': str(unified_file),
        'input_problems': str(benchmark_path),
        'model_name': model_path,
        'response_model': model_path,
        # Metadata for comprehensive analysis
        'benchmark_name': benchmark_name,
        'target_model_name': model_name,
    })

    # Initialize pipeline
    print(f"\n📋 Initializing pipeline...")
    try:
        pipeline = UnifiedBatchedPipeline(config)
    except Exception as e:
        print(f"❌ ERROR initializing pipeline: {e}")
        return None

    # Run all stages
    try:
        # Stage 1: Generate variations
        print(f"\n🔄 Stage 1: Generating variations...")
        pipeline.stage1_generate_variations_batched()

        # Stage 2: Validate variations
        print(f"\n✅ Stage 2: Validating variations...")
        pipeline.stage_validation()

        # Stage 3: Generate responses
        print(f"\n💬 Stage 3: Generating responses...")
        pipeline.stage2_generate_responses()

        # Stage 4: Evaluate drift
        print(f"\n📊 Stage 4: Evaluating drift...")
        pipeline.stage3_add_evaluation_metrics()

        # Save summary statistics
        stats = compute_experiment_stats(unified_file)
        stats_file = exp_dir / f"{experiment_id}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✅ EXPERIMENT COMPLETE: {experiment_id}")
        print(f"   Results: {unified_file}")
        print(f"   Stats: {stats_file}")
        print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return stats

    except Exception as e:
        print(f"\n❌ ERROR during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_experiment_stats(unified_file):
    """Compute summary statistics from experiment results."""

    with open(unified_file, 'r') as f:
        data = json.load(f)

    variants = [e for e in data if e.get('is_variant')]
    baselines = [e for e in data if e.get('is_baseline')]

    stats = {
        'total_problems': len(baselines),
        'total_variations': len(variants),
        'positive_drift_count': sum(1 for v in variants if v.get('positive_drift')),
        'negative_drift_count': sum(1 for v in variants if v.get('negative_drift')),
        'baseline_accuracy': sum(1 for b in baselines if b.get('baseline_matches_ground_truth')) / len(baselines) if baselines else 0,
        'variant_accuracy': sum(1 for v in variants if v.get('variant_matches_ground_truth')) / len(variants) if variants else 0,
    }

    # Compute positive drift rate (% of baseline failures recovered)
    baseline_failures = [b for b in baselines if not b.get('baseline_matches_ground_truth')]
    if baseline_failures:
        # Check which baseline failures have at least one successful variant
        recovered = 0
        for baseline in baseline_failures:
            problem_id = baseline.get('problem_id')
            problem_variants = [v for v in variants if v.get('problem_id') == problem_id]
            if any(v.get('variant_matches_ground_truth') for v in problem_variants):
                recovered += 1
        stats['positive_drift_rate'] = (recovered / len(baseline_failures)) * 100 if baseline_failures else 0
    else:
        stats['positive_drift_rate'] = 0

    # Variation type breakdown
    by_type = {}
    for v in variants:
        vtype = v.get('transformation_type', 'unknown')
        if vtype not in by_type:
            by_type[vtype] = {'count': 0, 'positive_drift': 0}
        by_type[vtype]['count'] += 1
        if v.get('positive_drift'):
            by_type[vtype]['positive_drift'] += 1
    stats['by_transformation_type'] = by_type

    return stats


def run_all_experiments(output_dir='experiments/paper_replication', models_subset=None, benchmarks_subset=None):
    """Run all model × benchmark experiments."""

    print("\n" + "="*80)
    print("BENCHDRIFT PAPER EXPERIMENT REPLICATION")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Models: {', '.join(MODELS.keys()) if not models_subset else ', '.join(models_subset)}")
    print(f"Benchmarks: {', '.join(BENCHMARKS.keys()) if not benchmarks_subset else ', '.join(benchmarks_subset)}")
    print(f"Variation types: ALL (generic + cluster + persona + long-context)")
    print(f"Client: VLLM (batched inference)")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Filter models/benchmarks if subsets specified
    models_to_run = {k: v for k, v in MODELS.items() if not models_subset or k in models_subset}
    benchmarks_to_run = {k: v for k, v in BENCHMARKS.items() if not benchmarks_subset or k in benchmarks_subset}

    total_experiments = len(models_to_run) * len(benchmarks_to_run)
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Estimated time: {total_experiments * 2:.0f}-{total_experiments * 4:.0f} hours (with VLLM batching)")

    # Run experiments
    all_stats = {}
    completed = 0
    failed = 0

    for model_name, model_path in models_to_run.items():
        for benchmark_name, benchmark_info in benchmarks_to_run.items():
            stats = run_single_experiment(
                model_name, model_path,
                benchmark_name, benchmark_info,
                output_dir
            )

            if stats:
                all_stats[f"{model_name}_{benchmark_name}"] = stats
                completed += 1
            else:
                failed += 1

    # Save overall summary
    summary_file = Path(output_dir) / 'experiment_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total_experiments,
        'completed': completed,
        'failed': failed,
        'experiments': all_stats
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*80)
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"="*80)
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Failed: {failed}")
    print(f"Summary: {summary_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return summary


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run BenchDrift paper experiments with VLLM batching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (5 models × 3 benchmarks = 15 runs)
  python run_paper_experiments.py

  # Run specific model only
  python run_paper_experiments.py --models phi-4

  # Run specific benchmark only
  python run_paper_experiments.py --benchmarks gsm8k

  # Run specific model × benchmark
  python run_paper_experiments.py --models phi-4 --benchmarks gsm8k

  # Custom output directory
  python run_paper_experiments.py --output experiments/test_run
        """
    )

    parser.add_argument('--output', default='experiments/paper_replication',
                       help='Output directory for experiment results')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                       help='Specific models to run (default: all)')
    parser.add_argument('--benchmarks', nargs='+', choices=list(BENCHMARKS.keys()),
                       help='Specific benchmarks to run (default: all)')

    args = parser.parse_args()

    # Check VLLM availability
    try:
        import vllm
        print("✅ VLLM available - will use batched inference")
    except ImportError:
        print("❌ ERROR: VLLM not installed!")
        print("   Install with: pip install vllm")
        print("   Or modify BASE_CONFIG to use client_type='rits' instead")
        sys.exit(1)

    # Run experiments
    run_all_experiments(
        output_dir=args.output,
        models_subset=args.models,
        benchmarks_subset=args.benchmarks
    )


if __name__ == '__main__':
    main()
