#!/usr/bin/env python3
"""
Compute Complexity Metrics for BenchDrift Variations
Addresses reviewer concern: "Positive drift = task simplification?"

Computes both linguistic and computational complexity to show
variations preserve complexity (format sensitivity, not simplification).
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

# Optional dependencies with graceful fallback
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except:
    print("Warning: spaCy not available. Parse depth analysis will be skipped.")
    SPACY_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except:
    print("Warning: textstat not available. Readability scores will be skipped.")
    TEXTSTAT_AVAILABLE = False


def compute_linguistic_complexity(text):
    """Compute linguistic complexity metrics."""
    complexity = {}

    # Word count
    words = text.split()
    complexity['word_count'] = len(words)

    # Sentence count (rough heuristic)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    complexity['sentence_count'] = len(sentences)

    # Average sentence length
    complexity['avg_sentence_length'] = complexity['word_count'] / max(complexity['sentence_count'], 1)

    # Lexical diversity (unique words / total words)
    unique_words = set(word.lower() for word in words if word.isalpha())
    complexity['lexical_diversity'] = len(unique_words) / max(len(words), 1)

    # Parse tree depth (if spaCy available)
    if SPACY_AVAILABLE:
        doc = nlp(text)
        depths = []
        for token in doc:
            # Calculate depth by counting ancestors
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            depths.append(depth)
        complexity['parse_depth_max'] = max(depths) if depths else 0
        complexity['parse_depth_avg'] = np.mean(depths) if depths else 0
    else:
        complexity['parse_depth_max'] = None
        complexity['parse_depth_avg'] = None

    # Readability scores (if textstat available)
    if TEXTSTAT_AVAILABLE:
        complexity['flesch_kincaid'] = textstat.flesch_kincaid_grade(text)
        complexity['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        complexity['automated_readability'] = textstat.automated_readability_index(text)
    else:
        complexity['flesch_kincaid'] = None
        complexity['flesch_reading_ease'] = None
        complexity['automated_readability'] = None

    return complexity


def compute_computational_complexity(text):
    """Compute computational/reasoning complexity metrics."""
    complexity = {}

    # Arithmetic operations
    ops_pattern = r'[+\-*/÷×%]'
    complexity['arithmetic_ops'] = len(re.findall(ops_pattern, text))

    # Numeric values (operands)
    nums_pattern = r'\b\d+\.?\d*\b'
    complexity['numeric_operands'] = len(re.findall(nums_pattern, text))

    # Inference step indicators
    step_indicators = r'\b(then|next|after|subsequently|therefore|thus|hence|consequently)\b'
    complexity['inference_steps'] = len(re.findall(step_indicators, text, re.IGNORECASE))

    # Conditionals
    conditionals = r'\b(if|when|unless|provided|suppose|given|assuming)\b'
    complexity['conditionals'] = len(re.findall(conditionals, text, re.IGNORECASE))

    # Comparisons
    comparisons = r'\b(more|less|greater|smaller|higher|lower|bigger|larger|than|before|after)\b|[<>=≤≥]'
    complexity['comparisons'] = len(re.findall(comparisons, text, re.IGNORECASE))

    # Logical connectors
    logical = r'\b(and|or|not|but|however|although|while)\b'
    complexity['logical_connectors'] = len(re.findall(logical, text, re.IGNORECASE))

    # Temporal expressions
    temporal = r'\b(day|week|month|year|hour|minute|second|morning|afternoon|evening|am|pm|before|after|during|until|since)\b'
    complexity['temporal_expressions'] = len(re.findall(temporal, text, re.IGNORECASE))

    # Weighted total complexity score
    complexity['total_complexity'] = (
        complexity['arithmetic_ops'] * 2 +
        complexity['numeric_operands'] * 1 +
        complexity['inference_steps'] * 3 +
        complexity['conditionals'] * 4 +
        complexity['comparisons'] * 2 +
        complexity['logical_connectors'] * 1 +
        complexity['temporal_expressions'] * 1
    )

    return complexity


def analyze_complexity_changes(unified_file, output_prefix):
    """Analyze complexity changes for all variations."""

    print(f"Loading data from {unified_file}...")
    with open(unified_file, 'r') as f:
        data = json.load(f)

    # Process all variants
    results = []
    baselines = {}

    # First pass: compute baseline complexities
    print("Computing baseline complexities...")
    for entry in data:
        if entry.get('is_baseline'):
            problem_id = entry['problem_id']
            text = entry.get('original_problem') or entry.get('baseline_problem', '')

            ling_comp = compute_linguistic_complexity(text)
            comp_comp = compute_computational_complexity(text)

            baselines[problem_id] = {
                'linguistic': ling_comp,
                'computational': comp_comp,
                'text': text
            }

    print(f"Processed {len(baselines)} baselines")

    # Second pass: compute variant complexities and deltas
    print("Computing variant complexities and deltas...")
    variants = [e for e in data if e.get('is_variant')]

    for i, variant in enumerate(variants):
        if i % 100 == 0:
            print(f"  Processing variant {i}/{len(variants)}...")

        problem_id = variant.get('problem_id', '')
        variation_text = variant.get('variation') or variant.get('modified_problem', '')

        if problem_id not in baselines:
            continue

        # Compute variant complexity
        ling_comp = compute_linguistic_complexity(variation_text)
        comp_comp = compute_computational_complexity(variation_text)

        # Compute deltas
        baseline_ling = baselines[problem_id]['linguistic']
        baseline_comp = baselines[problem_id]['computational']

        delta_ling = {}
        for key in ling_comp:
            if ling_comp[key] is not None and baseline_ling[key] is not None:
                delta_ling[f'delta_{key}'] = ling_comp[key] - baseline_ling[key]
            else:
                delta_ling[f'delta_{key}'] = None

        delta_comp = {}
        for key in comp_comp:
            delta_comp[f'delta_{key}'] = comp_comp[key] - baseline_comp[key]

        # Combine all metrics
        result = {
            'variation_id': variant.get('variation_id'),
            'problem_id': problem_id,
            'transformation_type': variant.get('transformation_type'),
            'debugging_capability': variant.get('debugging_capability'),
            'positive_drift': variant.get('positive_drift', False),
            'negative_drift': variant.get('negative_drift', False),
            'combination_size': variant.get('combination_size', 0),
            'cross_domain': variant.get('cross_domain', False),

            # Baseline complexity
            **{f'baseline_ling_{k}': v for k, v in baseline_ling.items()},
            **{f'baseline_comp_{k}': v for k, v in baseline_comp.items()},

            # Variant complexity
            **{f'variant_ling_{k}': v for k, v in ling_comp.items()},
            **{f'variant_comp_{k}': v for k, v in comp_comp.items()},

            # Deltas
            **delta_ling,
            **delta_comp,
        }

        results.append(result)

    # Save detailed results
    output_file = Path(output_prefix + '_complexity_detailed.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_file}")

    # Compute summary statistics
    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*70)

    analyze_complexity_stratification(results, output_prefix)
    analyze_complexity_by_transformation(results, output_prefix)
    analyze_drift_by_complexity_change(results, output_prefix)

    return results


def analyze_complexity_stratification(results, output_prefix):
    """Stratify positive drift by complexity change."""

    print("\n📊 STRATIFICATION BY COMPLEXITY CHANGE")

    # Use total computational complexity delta as primary metric
    complexity_neutral_threshold = 2.0  # Within ±2 points = neutral

    neutral = []
    reducing = []
    increasing = []

    for r in results:
        delta = r.get('delta_total_complexity', 0)
        if abs(delta) <= complexity_neutral_threshold:
            neutral.append(r)
        elif delta < -complexity_neutral_threshold:
            reducing.append(r)
        else:
            increasing.append(r)

    def compute_drift_rate(variants):
        if not variants:
            return 0.0, 0
        pos_drift = sum(1 for v in variants if v.get('positive_drift'))
        return (pos_drift / len(variants)) * 100, len(variants)

    neutral_rate, neutral_count = compute_drift_rate(neutral)
    reducing_rate, reducing_count = compute_drift_rate(reducing)
    increasing_rate, increasing_count = compute_drift_rate(increasing)

    print(f"\n  Complexity-Neutral (Δ ≤ {complexity_neutral_threshold}):")
    print(f"    Count: {neutral_count:,}")
    print(f"    Positive drift rate: {neutral_rate:.2f}%")

    print(f"\n  Complexity-Reducing (Δ < -{complexity_neutral_threshold}):")
    print(f"    Count: {reducing_count:,}")
    print(f"    Positive drift rate: {reducing_rate:.2f}%")

    print(f"\n  Complexity-Increasing (Δ > {complexity_neutral_threshold}):")
    print(f"    Count: {increasing_count:,}")
    print(f"    Positive drift rate: {increasing_rate:.2f}%")

    # Save stratification results
    stratification = {
        'threshold': complexity_neutral_threshold,
        'neutral': {'count': neutral_count, 'drift_rate': neutral_rate},
        'reducing': {'count': reducing_count, 'drift_rate': reducing_rate},
        'increasing': {'count': increasing_count, 'drift_rate': increasing_rate},
        'interpretation': f"{(neutral_count/len(results)*100):.1f}% of variations are complexity-neutral"
    }

    output_file = Path(output_prefix + '_complexity_stratification.json')
    with open(output_file, 'w') as f:
        json.dump(stratification, f, indent=2)
    print(f"\n  ✅ Stratification saved to {output_file}")


def analyze_complexity_by_transformation(results, output_prefix):
    """Analyze average complexity change by transformation type."""

    print("\n📈 COMPLEXITY CHANGE BY TRANSFORMATION TYPE")

    by_type = defaultdict(list)
    for r in results:
        trans_type = r.get('transformation_type', 'unknown')
        delta = r.get('delta_total_complexity', 0)
        by_type[trans_type].append(delta)

    type_stats = {}
    for trans_type, deltas in by_type.items():
        type_stats[trans_type] = {
            'mean_delta': np.mean(deltas),
            'std_delta': np.std(deltas),
            'median_delta': np.median(deltas),
            'count': len(deltas)
        }

    # Sort by absolute mean delta
    sorted_types = sorted(type_stats.items(), key=lambda x: abs(x[1]['mean_delta']), reverse=True)

    print(f"\n  Top 10 Transformation Types by Complexity Change:")
    for i, (trans_type, stats) in enumerate(sorted_types[:10], 1):
        print(f"    {i:2d}. {trans_type[:50]:50s}: Δ={stats['mean_delta']:+6.2f} ± {stats['std_delta']:5.2f} (n={stats['count']})")

    # Save
    output_file = Path(output_prefix + '_complexity_by_type.json')
    with open(output_file, 'w') as f:
        json.dump(type_stats, f, indent=2)
    print(f"\n  ✅ Saved to {output_file}")


def analyze_drift_by_complexity_change(results, output_prefix):
    """Correlate drift with complexity change."""

    print("\n🔍 DRIFT CORRELATION WITH COMPLEXITY CHANGE")

    # Positive drift cases
    pos_drift = [r for r in results if r.get('positive_drift')]
    no_drift = [r for r in results if not r.get('positive_drift') and not r.get('negative_drift')]
    neg_drift = [r for r in results if r.get('negative_drift')]

    def avg_complexity_delta(variants):
        deltas = [v.get('delta_total_complexity', 0) for v in variants]
        return np.mean(deltas) if deltas else 0.0

    pos_avg = avg_complexity_delta(pos_drift)
    no_avg = avg_complexity_delta(no_drift)
    neg_avg = avg_complexity_delta(neg_drift)

    print(f"\n  Average Complexity Change:")
    print(f"    Positive drift cases: Δ = {pos_avg:+.2f}")
    print(f"    No drift cases:      Δ = {no_avg:+.2f}")
    print(f"    Negative drift cases: Δ = {neg_avg:+.2f}")

    # Point-biserial correlation
    try:
        from scipy.stats import pointbiserial
        deltas = [r.get('delta_total_complexity', 0) for r in results]
        drifts = [1 if r.get('positive_drift') else 0 for r in results]
        corr, p_value = pointbiserial(drifts, deltas)
        print(f"\n  Point-biserial correlation (positive drift ~ complexity change):")
        print(f"    r = {corr:.3f}, p = {p_value:.4f}")
    except ImportError:
        print("\n  (scipy not available for correlation test)")

    correlation_results = {
        'positive_drift_avg_delta': pos_avg,
        'no_drift_avg_delta': no_avg,
        'negative_drift_avg_delta': neg_avg,
        'positive_drift_count': len(pos_drift),
        'no_drift_count': len(no_drift),
        'negative_drift_count': len(neg_drift)
    }

    output_file = Path(output_prefix + '_drift_complexity_correlation.json')
    with open(output_file, 'w') as f:
        json.dump(correlation_results, f, indent=2)
    print(f"\n  ✅ Saved to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_complexity_metrics.py <unified_results.json> [output_prefix]")
        print("\nExample:")
        print("  python compute_complexity_metrics.py data/gsm8k_results.json data/gsm8k")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    output_prefix = sys.argv[2] if len(sys.argv) > 2 else input_file.stem

    print("="*70)
    print("BENCHDRIFT COMPLEXITY ANALYSIS")
    print("="*70)
    print(f"\nInput: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print()

    results = analyze_complexity_changes(input_file, output_prefix)

    print("\n" + "="*70)
    print(f"✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {output_prefix}_complexity_detailed.json")
    print(f"  - {output_prefix}_complexity_stratification.json")
    print(f"  - {output_prefix}_complexity_by_type.json")
    print(f"  - {output_prefix}_drift_complexity_correlation.json")
    print()
