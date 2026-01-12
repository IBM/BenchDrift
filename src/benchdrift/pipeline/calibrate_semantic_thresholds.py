#!/usr/bin/env python3
"""
Calibrate Semantic Clustering Thresholds

Automatically finds optimal thresholds for semantic clustering by:
1. Testing different threshold combinations on a sample
2. Measuring clustering quality metrics
3. Identifying diminishing returns point
4. Recommending production-ready thresholds

Usage:
    python calibrate_semantic_thresholds.py \
        --input problems.json \
        --sample-size 50 \
        --output calibration_results.json
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from collections import Counter

from benchdrift.pipeline.semantic_composite_detector import SemanticCompositeDetector, Candidate



@dataclass
class ThresholdConfig:
    """Configuration for threshold testing."""
    semantic_threshold: float
    spatial_threshold: int
    contiguous_gap: int


@dataclass
class ClusteringMetrics:
    """Metrics for evaluating clustering quality."""
    num_problems: int
    total_candidates: int
    total_segments: int
    avg_segments_per_problem: float
    avg_segment_size: float
    singleton_ratio: float  # Ratio of 1-member segments
    large_segment_ratio: float  # Ratio of segments with 5+ members
    segment_size_std: float  # Standard deviation of segment sizes
    coherence_score: float  # Semantic coherence within segments
    separation_score: float  # Separation between segments
    quality_score: float  # Overall quality (higher = better)


class SemanticThresholdCalibrator:
    """Calibrate semantic clustering thresholds automatically."""

    def __init__(self, problems: List[Dict]):
        """
        Args:
            problems: List of problem dictionaries
        """
        self.problems = problems
        self.results = []

    def calibrate(self,
                  semantic_range: Tuple[float, float] = (0.4, 0.9),
                  spatial_range: Tuple[int, int] = (15, 60),
                  gap_range: Tuple[int, int] = (3, 10),
                  num_points: int = 5) -> Dict:
        """
        Test threshold combinations and find optimal values.

        Args:
            semantic_range: (min, max) for semantic_threshold
            spatial_range: (min, max) for spatial_threshold
            gap_range: (min, max) for contiguous_gap
            num_points: Number of points to test in each range

        Returns:
            Calibration results with recommended thresholds
        """
        print(f"\n{'='*80}")
        print(f"SEMANTIC CLUSTERING THRESHOLD CALIBRATION")
        print(f"{'='*80}\n")

        print(f"📊 Testing on {len(self.problems)} problems")
        print(f"   Semantic threshold range: {semantic_range}")
        print(f"   Spatial threshold range: {spatial_range}")
        print(f"   Contiguous gap range: {gap_range}")
        print(f"   Points per dimension: {num_points}")

        # Generate test configurations
        semantic_values = np.linspace(semantic_range[0], semantic_range[1], num_points)
        spatial_values = np.linspace(spatial_range[0], spatial_range[1], num_points, dtype=int)
        gap_values = np.linspace(gap_range[0], gap_range[1], num_points, dtype=int)

        total_tests = len(semantic_values) * len(spatial_values) * len(gap_values)
        print(f"\n🔬 Running {total_tests} threshold combinations...\n")

        test_count = 0
        best_config = None
        best_score = -np.inf

        for sem_thresh in semantic_values:
            for spat_thresh in spatial_values:
                for gap_thresh in gap_values:
                    test_count += 1

                    config = ThresholdConfig(
                        semantic_threshold=float(sem_thresh),
                        spatial_threshold=int(spat_thresh),
                        contiguous_gap=int(gap_thresh)
                    )

                    # Test this configuration
                    metrics = self._evaluate_thresholds(config)

                    self.results.append({
                        'config': asdict(config),
                        'metrics': asdict(metrics)
                    })

                    # Track best
                    if metrics.quality_score > best_score:
                        best_score = metrics.quality_score
                        best_config = config

                    # Progress
                    if test_count % 10 == 0:
                        print(f"   Progress: {test_count}/{total_tests} "
                              f"(best score so far: {best_score:.3f})")

        print(f"\n✅ Calibration complete!\n")

        # Analyze results
        analysis = self._analyze_results(best_config)

        return analysis

    def _evaluate_thresholds(self, config: ThresholdConfig) -> ClusteringMetrics:
        """Evaluate clustering quality for given thresholds."""

        # Initialize detector with these thresholds
        detector = SemanticCompositeDetector(
            semantic_threshold=config.semantic_threshold,
            spatial_threshold=config.spatial_threshold,
            contiguous_gap=config.contiguous_gap
        )

        total_candidates = 0
        total_segments = 0
        segment_sizes = []
        coherence_scores = []

        for problem in self.problems:
            problem_text = problem.get('problem', problem.get('question', ''))

            # Detect candidates (simplified - just split on words for speed)
            candidates = self._quick_candidate_detection(problem_text)
            total_candidates += len(candidates)

            if not candidates:
                continue

            # Detect composites
            try:
                segments, linked_groups = detector.detect_composites(problem_text, candidates)

                total_segments += len(segments)

                # Collect segment sizes
                for seg in segments:
                    size = len(seg.members) if hasattr(seg, 'members') else 1
                    segment_sizes.append(size)

                # Measure coherence (simplified)
                if segments:
                    coherence = self._measure_coherence(segments, detector)
                    coherence_scores.append(coherence)

            except Exception:
                # Skip problematic cases
                continue

        # Calculate metrics
        num_problems = len(self.problems)
        avg_segments = total_segments / max(num_problems, 1)

        if segment_sizes:
            avg_size = np.mean(segment_sizes)
            size_std = np.std(segment_sizes)
            singleton_ratio = sum(1 for s in segment_sizes if s == 1) / len(segment_sizes)
            large_ratio = sum(1 for s in segment_sizes if s >= 5) / len(segment_sizes)
        else:
            avg_size = 0
            size_std = 0
            singleton_ratio = 1.0
            large_ratio = 0

        coherence_score = np.mean(coherence_scores) if coherence_scores else 0

        # Separation score (based on segment count - not too many, not too few)
        # Target: 3-8 segments per problem
        target_segments = 5
        separation_score = 1.0 - abs(avg_segments - target_segments) / target_segments
        separation_score = max(0, separation_score)

        # Quality score (composite metric)
        # Good clustering has:
        # - Moderate number of segments (not too many, not too few)
        # - Not too many singletons (low singleton_ratio)
        # - Some larger segments (moderate large_ratio)
        # - High coherence within segments
        # - Good separation

        quality_score = (
            (1 - singleton_ratio) * 0.3 +  # Penalize too many singletons
            large_ratio * 0.2 +  # Reward some large segments
            coherence_score * 0.3 +  # Reward coherent segments
            separation_score * 0.2  # Reward good segment count
        )

        return ClusteringMetrics(
            num_problems=num_problems,
            total_candidates=total_candidates,
            total_segments=total_segments,
            avg_segments_per_problem=avg_segments,
            avg_segment_size=avg_size,
            singleton_ratio=singleton_ratio,
            large_segment_ratio=large_ratio,
            segment_size_std=size_std,
            coherence_score=coherence_score,
            separation_score=separation_score,
            quality_score=quality_score
        )

    def _quick_candidate_detection(self, text: str) -> List[Candidate]:
        """Quick candidate detection for calibration."""
        import re

        candidates = []

        # Numbers
        for match in re.finditer(r'-?\d+\.?\d*', text):
            candidates.append(Candidate(
                text=match.group(),
                span=(match.start(), match.end()),
                type='number'
            ))

        # Words (3+ characters)
        for match in re.finditer(r'\b[a-zA-Z]{3,}\b', text):
            candidates.append(Candidate(
                text=match.group(),
                span=(match.start(), match.end()),
                type='word'
            ))

        return candidates

    def _measure_coherence(self, segments, detector) -> float:
        """Measure semantic coherence within segments."""
        if not segments:
            return 0

        coherence_scores = []

        for seg in segments:
            members = seg.members if hasattr(seg, 'members') else []
            if len(members) < 2:
                continue

            # Get embeddings for members
            texts = [m.text for m in members]
            try:
                embeddings = detector.embedder.encode(texts)

                # Calculate pairwise similarities
                from scipy.spatial.distance import cosine
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = 1 - cosine(embeddings[i], embeddings[j])
                        similarities.append(sim)

                if similarities:
                    coherence_scores.append(np.mean(similarities))
            except Exception:
                continue

        return np.mean(coherence_scores) if coherence_scores else 0

    def _analyze_results(self, best_config: ThresholdConfig) -> Dict:
        """Analyze calibration results and generate recommendations."""

        print(f"{'='*80}")
        print(f"CALIBRATION RESULTS")
        print(f"{'='*80}\n")

        # Find best configuration
        best_result = max(self.results, key=lambda x: x['metrics']['quality_score'])
        best_metrics = best_result['metrics']

        print(f"🎯 RECOMMENDED THRESHOLDS:")
        print(f"   --semantic-threshold {best_config.semantic_threshold:.2f}")
        print(f"   --spatial-threshold {best_config.spatial_threshold}")
        print(f"   --contiguous-gap {best_config.contiguous_gap}")

        print(f"\n📊 CLUSTERING QUALITY METRICS:")
        print(f"   Quality score: {best_metrics['quality_score']:.3f}")
        print(f"   Avg segments per problem: {best_metrics['avg_segments_per_problem']:.1f}")
        print(f"   Avg segment size: {best_metrics['avg_segment_size']:.1f}")
        print(f"   Singleton ratio: {best_metrics['singleton_ratio']:.1%}")
        print(f"   Large segment ratio: {best_metrics['large_segment_ratio']:.1%}")
        print(f"   Coherence score: {best_metrics['coherence_score']:.3f}")
        print(f"   Separation score: {best_metrics['separation_score']:.3f}")

        # Analyze sensitivity
        print(f"\n🔍 SENSITIVITY ANALYSIS:")
        self._plot_sensitivity_analysis()

        # Diminishing returns analysis
        print(f"\n📉 DIMINISHING RETURNS ANALYSIS:")
        self._analyze_diminishing_returns()

        return {
            'recommended_config': asdict(best_config),
            'best_metrics': best_metrics,
            'all_results': self.results,
            'interpretation': self._interpret_results(best_metrics)
        }

    def _plot_sensitivity_analysis(self):
        """Plot how metrics vary with thresholds."""

        # Extract data
        semantic_vals = [r['config']['semantic_threshold'] for r in self.results]
        spatial_vals = [r['config']['spatial_threshold'] for r in self.results]
        quality_vals = [r['metrics']['quality_score'] for r in self.results]
        segment_counts = [r['metrics']['avg_segments_per_problem'] for r in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Threshold Sensitivity Analysis', fontsize=16, fontweight='bold')

        # Panel 1: Semantic threshold vs Quality
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(semantic_vals, quality_vals, c=spatial_vals,
                              cmap='viridis', alpha=0.6, s=50)
        ax1.set_xlabel('Semantic Threshold', fontweight='bold')
        ax1.set_ylabel('Quality Score', fontweight='bold')
        ax1.set_title('Semantic Threshold Impact')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Spatial Threshold')

        # Panel 2: Spatial threshold vs Segment Count
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(spatial_vals, segment_counts, c=semantic_vals,
                              cmap='plasma', alpha=0.6, s=50)
        ax2.set_xlabel('Spatial Threshold', fontweight='bold')
        ax2.set_ylabel('Avg Segments per Problem', fontweight='bold')
        ax2.set_title('Spatial Threshold Impact')
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Semantic Threshold')

        # Panel 3: Quality vs Segment Count
        ax3 = axes[1, 0]
        ax3.scatter(segment_counts, quality_vals, alpha=0.6, s=50, color='#e74c3c')
        ax3.set_xlabel('Avg Segments per Problem', fontweight='bold')
        ax3.set_ylabel('Quality Score', fontweight='bold')
        ax3.set_title('Segment Count vs Quality')
        ax3.grid(alpha=0.3)

        # Panel 4: Pareto frontier (Quality vs Complexity)
        ax4 = axes[1, 1]
        singleton_ratios = [r['metrics']['singleton_ratio'] for r in self.results]
        ax4.scatter(singleton_ratios, quality_vals, alpha=0.6, s=50, color='#2ecc71')
        ax4.set_xlabel('Singleton Ratio (Lower = Better)', fontweight='bold')
        ax4.set_ylabel('Quality Score', fontweight='bold')
        ax4.set_title('Quality vs Fragmentation')
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Saved sensitivity analysis to: threshold_sensitivity_analysis.png")

    def _analyze_diminishing_returns(self):
        """Identify point of diminishing returns."""

        # Sort by quality score
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['quality_score'])

        # Calculate marginal improvements
        quality_scores = [r['metrics']['quality_score'] for r in sorted_results]

        # Find elbow point (where improvement drops below threshold)
        improvements = np.diff(quality_scores)

        # When improvement drops below 1% of max, we've hit diminishing returns
        max_improvement = max(improvements) if len(improvements) > 0 else 0
        threshold = 0.01 * max_improvement

        diminishing_idx = None
        for i, imp in enumerate(improvements):
            if imp < threshold:
                diminishing_idx = i
                break

        if diminishing_idx:
            cutoff_score = quality_scores[diminishing_idx]
            good_configs = [r for r in self.results
                           if r['metrics']['quality_score'] >= cutoff_score]

            print(f"   Diminishing returns point: quality score > {cutoff_score:.3f}")
            print(f"   Configs above threshold: {len(good_configs)}/{len(self.results)}")
            print(f"   → Further optimization beyond this point yields < 1% improvement")
        else:
            print(f"   No clear diminishing returns point found")
            print(f"   → All tested configurations show meaningful differences")

    def _interpret_results(self, metrics: Dict) -> Dict:
        """Interpret metrics and provide recommendations."""

        interpretation = {}

        # Segment count interpretation
        avg_segs = metrics['avg_segments_per_problem']
        if avg_segs < 3:
            interpretation['segment_count'] = "Too few segments - thresholds may be too loose. Consider decreasing semantic_threshold or spatial_threshold."
        elif avg_segs > 10:
            interpretation['segment_count'] = "Too many segments - thresholds may be too strict. Consider increasing semantic_threshold or spatial_threshold."
        else:
            interpretation['segment_count'] = "Good segment count - meaningful composites without over-fragmentation."

        # Singleton interpretation
        singleton_ratio = metrics['singleton_ratio']
        if singleton_ratio > 0.5:
            interpretation['singletons'] = "High singleton ratio - many candidates not clustering. This is acceptable if candidates are truly independent."
        else:
            interpretation['singletons'] = "Good clustering - most candidates finding related groups."

        # Quality interpretation
        quality = metrics['quality_score']
        if quality > 0.6:
            interpretation['overall'] = "Excellent clustering quality - recommended for production use."
        elif quality > 0.4:
            interpretation['overall'] = "Good clustering quality - acceptable for production, monitor results."
        else:
            interpretation['overall'] = "Low clustering quality - consider adjusting thresholds or reviewing data characteristics."

        return interpretation


def main():
    """CLI interface for threshold calibration."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Calibrate semantic clustering thresholds automatically'
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input problems file (JSON or JSONL)')
    parser.add_argument('--output', type=str, default='calibration_results.json',
                       help='Output calibration results file')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of problems to use for calibration (default: 50)')
    parser.add_argument('--num-points', type=int, default=5,
                       help='Number of points to test per dimension (default: 5)')

    # Threshold ranges
    parser.add_argument('--semantic-min', type=float, default=0.4,
                       help='Minimum semantic threshold to test (default: 0.4)')
    parser.add_argument('--semantic-max', type=float, default=0.9,
                       help='Maximum semantic threshold to test (default: 0.9)')
    parser.add_argument('--spatial-min', type=int, default=15,
                       help='Minimum spatial threshold to test (default: 15)')
    parser.add_argument('--spatial-max', type=int, default=60,
                       help='Maximum spatial threshold to test (default: 60)')
    parser.add_argument('--gap-min', type=int, default=3,
                       help='Minimum contiguous gap to test (default: 3)')
    parser.add_argument('--gap-max', type=int, default=10,
                       help='Maximum contiguous gap to test (default: 10)')

    args = parser.parse_args()

    # Load problems
    with open(args.input, 'r') as f:
        if args.input.endswith('.jsonl'):
            problems = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            problems = [data] if isinstance(data, dict) else data

    # Sample if needed
    if len(problems) > args.sample_size:
        import random
        problems = random.sample(problems, args.sample_size)
        print(f"Sampled {args.sample_size} problems from {len(problems)} total")

    # Run calibration
    calibrator = SemanticThresholdCalibrator(problems)

    results = calibrator.calibrate(
        semantic_range=(args.semantic_min, args.semantic_max),
        spatial_range=(args.spatial_min, args.spatial_max),
        gap_range=(args.gap_min, args.gap_max),
        num_points=args.num_points
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Calibration results saved to: {args.output}")

    # Print final recommendations
    print(f"\n{'='*80}")
    print(f"FINAL RECOMMENDATIONS")
    print(f"{'='*80}\n")

    config = results['recommended_config']
    interpretation = results['interpretation']

    print(f"✅ Use these thresholds for your full dataset:")
    print(f"\npython semantic_batched_pipeline.py \\")
    print(f"  --unified-file results.json \\")
    print(f"  --input your_problems.json \\")
    print(f"  --target-model phi-4 \\")
    print(f"  --semantic-threshold {config['semantic_threshold']:.2f} \\")
    print(f"  --spatial-threshold {config['spatial_threshold']} \\")
    print(f"  --contiguous-gap {config['contiguous_gap']}")

    print(f"\n💡 Interpretation:")
    for key, value in interpretation.items():
        print(f"   • {key}: {value}")

    print()


if __name__ == '__main__':
    main()
