"""
Comprehensive Results Visualizer for Variation Pipeline
Publication-quality visualizations showing model robustness testing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


class ComprehensiveResultsVisualizer:
    """Creates publication-quality result visualizations for research papers"""

    def __init__(self, df):
        """
        Initialize with dataframe containing variation data

        Args:
            df: DataFrame with columns for variations, drift, transformation types
        """
        self.df = df
        self.variants_df = df[df['is_variant'].fillna(False)].copy()

        # Fix transformation type prefixes (handle both old and new)
        self._normalize_transformation_types()

    def _normalize_transformation_types(self):
        """Normalize transformation type names for consistent handling"""
        # Handle both bucket_combination and direct_combination
        if 'transformation_type' in self.df.columns:
            self.df['transformation_type'] = self.df['transformation_type'].fillna('unknown')

            # Normalize to consistent naming
            self.df['trans_category'] = 'Other'
            self.df.loc[self.df['transformation_type'].str.startswith('generic_', na=False), 'trans_category'] = 'Generic'
            self.df.loc[self.df['transformation_type'].str.startswith('persona_', na=False), 'trans_category'] = 'Persona'
            self.df.loc[self.df['transformation_type'].str.startswith('long_context.', na=False), 'trans_category'] = 'Long Context'

            # Handle both bucket and direct combinations -> call them Clusters
            combo_mask = (self.df['transformation_type'].str.startswith('bucket_combination_', na=False) |
                         self.df['transformation_type'].str.startswith('direct_combination_', na=False))
            self.df.loc[combo_mask, 'trans_category'] = 'Cluster'

    @staticmethod
    def _clean_transformation_name(trans_type):
        """Clean transformation type name for display"""
        # Handle cluster variations (bucket/direct combinations)
        if 'combination_' in trans_type:
            # Extract the N from "bucket_combination_Nway" or "direct_combination_Nway"
            import re
            match = re.search(r'(\d+)way', trans_type)
            if match:
                n = match.group(1)
                return f'Cluster {n}'
            return 'Cluster'

        # Remove all prefixes and clean up
        cleaned = (trans_type
                .replace('cross_batch_generic_', '')
                .replace('cross_batch_', '')
                .replace('generic_', '')
                .replace('persona_', '')
                .replace('long_context.', '')
                .replace('.', ' ')
                .replace('_', ' ')
                .strip()
                .title())

        # Remove "Long Context" if it appears at the beginning after title casing
        if cleaned.startswith('Long Context '):
            cleaned = cleaned[13:]  # Remove "Long Context " (13 characters)

        return cleaned

    def create_comprehensive_summary(self, save_fig=False):
        """Create comprehensive research-quality summary with multiple panels"""
        import os
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

        # Create 2x2 grid
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Drift Overview (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_drift_overview(ax1)

        # Panel 2: Top Problematic Variations (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_top_problematic_variations(ax2)

        # Panel 3: Negative Drift Breakdown (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_drift_breakdown(ax3, drift_type='negative')

        # Panel 4: Positive Drift Breakdown (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_drift_breakdown(ax4, drift_type='positive')

        plt.suptitle('Model Robustness Analysis: Variation Testing Results',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/drift_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()

        # Print comprehensive text summary
        self._print_comprehensive_summary()

    def _plot_drift_overview(self, ax):
        """Plot overall drift statistics from baseline"""
        variants = self.df[self.df['is_variant']].copy()

        if 'has_drift' not in variants.columns:
            ax.text(0.5, 0.5, 'No drift data available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Ensure boolean columns
        variants['has_drift'] = variants['has_drift'].astype(bool)
        if 'has_improvement' in variants.columns:
            variants['has_improvement'] = variants['has_improvement'].astype(bool)
        else:
            variants['has_improvement'] = False

        # Calculate statistics
        total = len(variants)
        negative_drift = (variants['has_drift'] & ~variants['has_improvement']).sum()
        positive_drift = variants['has_improvement'].sum()
        no_drift = (~variants['has_drift']).sum()

        # Calculate percentages
        neg_pct = (negative_drift / total) * 100
        pos_pct = (positive_drift / total) * 100
        stable_pct = (no_drift / total) * 100

        # Create stacked bar chart
        categories = ['Variations']
        stable_vals = [stable_pct]
        pos_vals = [pos_pct]
        neg_vals = [neg_pct]

        x = np.arange(len(categories))
        width = 0.6

        # Stack bars
        p1 = ax.barh(x, stable_vals, width, label='Stable (No Drift)', color='#70AD47')
        p2 = ax.barh(x, pos_vals, width, left=stable_vals, label='Positive Drift', color='#2E8B57')
        p3 = ax.barh(x, neg_vals, width, left=[s+p for s,p in zip(stable_vals, pos_vals)],
                    label='Negative Drift', color='#E74C3C')

        # Add percentage labels
        if stable_pct > 5:
            ax.text(stable_pct/2, 0, f'{stable_pct:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=12)
        if pos_pct > 5:
            ax.text(stable_pct + pos_pct/2, 0, f'{pos_pct:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        if neg_pct > 5:
            ax.text(stable_pct + pos_pct + neg_pct/2, 0, f'{neg_pct:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=12, color='white')

        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage of Variations (%)', fontweight='bold')
        ax.set_title('Overall Drift from Baseline Performance', fontweight='bold', fontsize=13)
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

        # Add count labels
        ax.text(0.5, 0.95, f'Total: {total} variations | Neg: {negative_drift} | Pos: {positive_drift} | Stable: {no_drift}',
               transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    def _plot_top_problematic_variations(self, ax):
        """Plot top 5 variation types with highest drift rates"""
        variants = self.df[self.df['is_variant']].copy()

        if 'has_drift' not in variants.columns or 'transformation_type' not in variants.columns:
            ax.text(0.5, 0.5, 'No variation type data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate drift rate per transformation type
        # Ensure has_drift is boolean
        variants['has_drift'] = variants['has_drift'].astype(bool)

        drift_by_type = variants.groupby('transformation_type').agg({
            'has_drift': ['sum', 'count', 'mean']
        }).reset_index()

        drift_by_type.columns = ['transformation_type', 'drift_count', 'total_count', 'drift_rate']

        # Ensure numeric types
        drift_by_type['drift_count'] = pd.to_numeric(drift_by_type['drift_count'], errors='coerce').fillna(0)
        drift_by_type['total_count'] = pd.to_numeric(drift_by_type['total_count'], errors='coerce').fillna(0)
        drift_by_type['drift_rate'] = pd.to_numeric(drift_by_type['drift_rate'], errors='coerce').fillna(0)

        drift_by_type = drift_by_type[drift_by_type['total_count'] >= 3]  # At least 3 samples
        drift_by_type = drift_by_type.sort_values('drift_rate', ascending=False).head(5)

        if len(drift_by_type) == 0:
            ax.text(0.5, 0.5, 'Insufficient data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Clean transformation type names
        drift_by_type['clean_name'] = drift_by_type['transformation_type'].apply(
            lambda x: self._clean_transformation_name(x)[:40]
        )

        # Create horizontal bar chart
        y_pos = np.arange(len(drift_by_type))
        colors = plt.cm.Reds(drift_by_type['drift_rate'].values)

        bars = ax.barh(y_pos, drift_by_type['drift_rate'] * 100, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(drift_by_type['clean_name'], fontsize=10)
        ax.set_xlabel('Drift Rate (%)', fontweight='bold')
        ax.set_title('Top 5 Most Problematic Variation Types', fontweight='bold', fontsize=13)
        ax.set_xlim(0, 100)

        # Add value labels
        for i, (idx, row) in enumerate(drift_by_type.iterrows()):
            ax.text(row['drift_rate'] * 100 + 2, i,
                   f"{row['drift_rate']*100:.1f}% ({int(row['drift_count'])}/{int(row['total_count'])})",
                   va='center', fontsize=9)

    def _plot_drift_breakdown(self, ax, drift_type='negative'):
        """Plot top 5 variation types by drift category (negative or positive)"""
        variants = self.df[self.df['is_variant']].copy()

        if 'has_drift' not in variants.columns:
            ax.text(0.5, 0.5, f'No {drift_type} drift data',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Ensure boolean columns
        variants['has_drift'] = variants['has_drift'].astype(bool)
        if 'has_improvement' in variants.columns:
            variants['has_improvement'] = variants['has_improvement'].astype(bool)
        else:
            variants['has_improvement'] = False

        # Filter by drift type
        if drift_type == 'negative':
            drift_variants = variants[variants['has_drift'] & ~variants['has_improvement']].copy()
            title = 'Top 5 Variation Types: Negative Drift'
            color_map = plt.cm.Reds
        else:
            if 'has_improvement' not in variants.columns:
                ax.text(0.5, 0.5, 'No positive drift data',
                       ha='center', va='center', transform=ax.transAxes)
                return
            drift_variants = variants[variants['has_improvement']].copy()
            title = 'Top 5 Variation Types: Positive Drift'
            color_map = plt.cm.Greens

        if len(drift_variants) == 0:
            ax.text(0.5, 0.5, f'No {drift_type} drift detected',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold', fontsize=13)
            return

        # Count by transformation type
        type_counts = drift_variants['transformation_type'].value_counts().head(5)

        # Clean names
        clean_names = [
            t.replace('generic_', '').replace('persona_', '').replace('bucket_combination_', 'Combo-')
             .replace('direct_combination_', 'Combo-').replace('_', ' ').title()[:30]
            for t in type_counts.index
        ]

        # Calculate percentages
        total_drift = len(drift_variants)
        percentages = (type_counts.values / total_drift) * 100

        # Create horizontal bar chart
        y_pos = np.arange(len(type_counts))
        colors = color_map(np.linspace(0.5, 0.9, len(type_counts)))

        bars = ax.barh(y_pos, percentages, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_names, fontsize=10)
        ax.set_xlabel(f'Percentage of {drift_type.title()} Drift Cases (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=13)

        # Add value labels
        for i, (count, pct) in enumerate(zip(type_counts.values, percentages)):
            ax.text(pct + 1, i, f'{pct:.1f}% (n={count})', va='center', fontsize=9)

        ax.set_xlim(0, max(percentages) * 1.2)

    def _print_comprehensive_summary(self):
        """Print detailed text summary of results"""
        variants = self.df[self.df['is_variant']].copy()
        problems_count = self.df['problem_id'].nunique()
        variants_count = len(variants)

        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ROBUSTNESS ANALYSIS SUMMARY")
        print("="*80)

        # Dataset overview
        print("\n📋 Dataset Overview:")
        print(f"   • Total Problems: {problems_count}")
        print(f"   • Total Variations: {variants_count}")
        print(f"   • Variations per Problem: {variants_count/problems_count:.1f} avg")

        if 'has_drift' in variants.columns:
            # Ensure boolean columns
            variants['has_drift'] = variants['has_drift'].astype(bool)
            if 'has_improvement' in variants.columns:
                variants['has_improvement'] = variants['has_improvement'].astype(bool)
            else:
                variants['has_improvement'] = False

            # Calculate drift statistics
            total = len(variants)
            negative_drift = (variants['has_drift'] & ~variants['has_improvement']).sum()
            positive_drift = variants['has_improvement'].sum()
            no_drift = (~variants['has_drift']).sum()

            neg_pct = (negative_drift / total) * 100
            pos_pct = (positive_drift / total) * 100
            stable_pct = (no_drift / total) * 100

            # Overall drift analysis
            print("\n🎯 Overall Drift Analysis:")
            print(f"   • Stable (No Drift): {no_drift} ({stable_pct:.1f}%)")
            print(f"   • Negative Drift: {negative_drift} ({neg_pct:.1f}%)")
            print(f"   • Positive Drift: {positive_drift} ({pos_pct:.1f}%)")

            # Drift rate assessment
            total_drift_rate = ((negative_drift + positive_drift) / total) * 100
            if total_drift_rate > 20:
                status = "🔴 HIGH"
            elif total_drift_rate > 10:
                status = "🟡 MEDIUM"
            else:
                status = "🟢 LOW"
            print(f"   • Total Drift Rate: {total_drift_rate:.1f}% {status}")

            # Top problematic variations
            print("\n⚠️  Top 5 Most Problematic Variation Types:")
            drift_by_type = variants.groupby('transformation_type').agg({
                'has_drift': ['sum', 'count', 'mean']
            }).reset_index()
            drift_by_type.columns = ['transformation_type', 'drift_count', 'total_count', 'drift_rate']
            drift_by_type = drift_by_type[drift_by_type['total_count'] >= 3]
            top_problematic = drift_by_type.sort_values('drift_rate', ascending=False).head(5)

            for idx, row in top_problematic.iterrows():
                clean_name = self._clean_transformation_name(row['transformation_type'])
                print(f"   {idx+1}. {clean_name}: {row['drift_rate']*100:.1f}% "
                     f"({int(row['drift_count'])}/{int(row['total_count'])} cases)")

            # Negative drift details
            if negative_drift > 0:
                print("\n🔴 Negative Drift Breakdown (Top 5):")
                neg_variants = variants[variants['has_drift'] & ~variants['has_improvement']]
                neg_counts = neg_variants['transformation_type'].value_counts().head(5)
                for i, (trans_type, count) in enumerate(neg_counts.items(), 1):
                    clean_name = self._clean_transformation_name(trans_type)
                    pct = (count / negative_drift) * 100
                    print(f"   {i}. {clean_name}: {count} cases ({pct:.1f}% of neg. drift)")

            # Positive drift details
            if positive_drift > 0:
                print("\n🟢 Positive Drift Breakdown (Top 5):")
                pos_variants = variants[variants['has_improvement']]
                pos_counts = pos_variants['transformation_type'].value_counts().head(5)
                for i, (trans_type, count) in enumerate(pos_counts.items(), 1):
                    clean_name = self._clean_transformation_name(trans_type)
                    pct = (count / positive_drift) * 100
                    print(f"   {i}. {clean_name}: {count} cases ({pct:.1f}% of pos. drift)")

            # Variation category analysis
            print("\n📂 Variation Category Analysis:")
            cat_stats = variants.groupby('trans_category').agg({
                'has_drift': ['sum', 'count', 'mean']
            })
            cat_stats.columns = ['drift_count', 'total', 'drift_rate']

            for category in ['Generic', 'Combination', 'Persona', 'Other']:
                if category in cat_stats.index:
                    stats = cat_stats.loc[category]
                    print(f"   • {category}: {int(stats['total'])} variations, "
                         f"{stats['drift_rate']*100:.1f}% drift rate "
                         f"({int(stats['drift_count'])} cases)")

        print("\n" + "="*80)
        print("✅ Analysis Complete - Use visualizations above for publication/presentation")
        print("="*80 + "\n")

    def show_drift_examples(self, top_n=3):
        """
        Show concrete examples of positive and negative drift cases.
        For top N variation types, shows ground truth, baseline answer, and variant answer.
        """
        variants = self.df[self.df['is_variant'] == True].copy()

        if 'has_drift' not in variants.columns:
            print("⚠️  No drift data available for examples")
            return

        print("\n" + "="*80)
        print("📋 CONCRETE DRIFT EXAMPLES")
        print("="*80)

        # === NEGATIVE DRIFT EXAMPLES ===
        neg_variants = variants[variants['has_drift'] & ~variants.get('has_improvement', False)].copy()

        if len(neg_variants) > 0:
            print("\n" + "🔴"*40)
            print("NEGATIVE DRIFT EXAMPLES (Model Performance Degraded)")
            print("🔴"*40 + "\n")

            # Get top N transformation types by count
            neg_type_counts = neg_variants['transformation_type'].value_counts().head(top_n)

            for rank, (trans_type, count) in enumerate(neg_type_counts.items(), 1):
                # Get one example from this type
                example = neg_variants[neg_variants['transformation_type'] == trans_type].iloc[0]

                clean_type = self._clean_transformation_name(trans_type)

                print(f"{'─'*80}")
                print(f"Example {rank}: {clean_type}")
                print(f"{'─'*80}")
                print(f"\n📝 ORIGINAL PROBLEM:")
                print(f"   {example.get('original_problem', 'N/A')}")

                print(f"\n📝 VARIATION:")
                print(f"   {example.get('modified_problem', 'N/A')}")

                print(f"\n✅ GROUND TRUTH ANSWER:")
                print(f"   {example.get('ground_truth_answer', 'N/A')}")

                # Find baseline answer for this problem
                baseline = self.df[(self.df['is_baseline']) &
                                  (self.df['problem_id'] == example.get('problem_id'))].iloc[0] if len(self.df[self.df['is_baseline']]) > 0 else None

                if baseline is not None:
                    baseline_ans = baseline.get('baseline_answer') or baseline.get('baseline_model_answer') or baseline.get('model_final_answer', 'N/A')
                    baseline_correct = baseline.get('baseline_matches_ground_truth', False)
                    print(f"\n✓ BASELINE ANSWER (Original Problem):")
                    print(f"   {baseline_ans}")
                    print(f"   Status: {'Correct' if baseline_correct else 'Incorrect'}")

                variant_ans = example.get('variant_answer') or example.get('model_final_answer', 'N/A')
                variant_correct = example.get('variant_matches_ground_truth', False)
                print(f"\n✗ VARIANT ANSWER (Negative Drift):")
                print(f"   {variant_ans}")
                print(f"   Status: {'Correct' if variant_correct else 'Incorrect'}")

                # Show drift details
                if 'baseline_correctness' in example and 'variant_correctness' in example:
                    print(f"\n📊 DRIFT DETAILS:")
                    print(f"   Baseline: {example['baseline_correctness']}")
                    print(f"   Variant: {example['variant_correctness']}")
                    print(f"   Change: {example['baseline_correctness']} → {example['variant_correctness']} (DEGRADATION)")

                print(f"\n💡 IMPACT: This variation type caused {count} negative drift cases")
                print()

        # === POSITIVE DRIFT EXAMPLES ===
        if 'has_improvement' in variants.columns:
            pos_variants = variants[variants['has_improvement']].copy()

            if len(pos_variants) > 0:
                print("\n" + "🟢"*40)
                print("POSITIVE DRIFT EXAMPLES (Model Performance Improved)")
                print("🟢"*40 + "\n")

                # Get top N transformation types by count
                pos_type_counts = pos_variants['transformation_type'].value_counts().head(top_n)

                for rank, (trans_type, count) in enumerate(pos_type_counts.items(), 1):
                    # Get one example from this type
                    example = pos_variants[pos_variants['transformation_type'] == trans_type].iloc[0]

                    clean_type = self._clean_transformation_name(trans_type)

                    print(f"{'─'*80}")
                    print(f"Example {rank}: {clean_type}")
                    print(f"{'─'*80}")
                    print(f"\n📝 ORIGINAL PROBLEM:")
                    print(f"   {example.get('original_problem', 'N/A')}")

                    print(f"\n📝 VARIATION:")
                    print(f"   {example.get('modified_problem', 'N/A')}")

                    print(f"\n✅ GROUND TRUTH ANSWER:")
                    print(f"   {example.get('ground_truth_answer', 'N/A')}")

                    # Find baseline answer for this problem
                    baseline = self.df[(self.df['is_baseline']) &
                                      (self.df['problem_id'] == example.get('problem_id'))].iloc[0] if len(self.df[self.df['is_baseline']]) > 0 else None

                    if baseline is not None:
                        baseline_ans = baseline.get('baseline_answer') or baseline.get('baseline_model_answer') or baseline.get('model_final_answer', 'N/A')
                        baseline_correct = baseline.get('baseline_matches_ground_truth', False)
                        print(f"\n✗ BASELINE ANSWER (Original Problem):")
                        print(f"   {baseline_ans}")
                        print(f"   Status: {'Correct' if baseline_correct else 'Incorrect'}")

                    variant_ans = example.get('variant_answer') or example.get('model_final_answer', 'N/A')
                    variant_correct = example.get('variant_matches_ground_truth', False)
                    print(f"\n✓ VARIANT ANSWER (Positive Drift):")
                    print(f"   {variant_ans}")
                    print(f"   Status: {'Correct' if variant_correct else 'Incorrect'}")

                    # Show drift details
                    if 'baseline_correctness' in example and 'variant_correctness' in example:
                        print(f"\n📊 DRIFT DETAILS:")
                        print(f"   Baseline: {example['baseline_correctness']}")
                        print(f"   Variant: {example['variant_correctness']}")
                        print(f"   Change: {example['baseline_correctness']} → {example['variant_correctness']} (IMPROVEMENT)")

                    print(f"\n💡 IMPACT: This variation type caused {count} positive drift cases")
                    print()

        print("="*80)
        print("✅ Concrete examples show how variations affect model performance")
        print("="*80 + "\n")


# Convenience function to use in notebooks
def visualize_results(df, show_examples=True, top_n_examples=3, save_fig=False):
    """
    Quick function to generate all visualizations

    Args:
        df: DataFrame with variation results
        show_examples: Whether to show concrete drift examples (default: True)
        top_n_examples: Number of example categories to show for pos/neg drift (default: 3)
        save_fig: Whether to save figure to figures/ directory (default: False)
    """
    viz = ComprehensiveResultsVisualizer(df)
    viz.create_comprehensive_summary(save_fig=save_fig)

    if show_examples:
        viz.show_drift_examples(top_n=top_n_examples)

    return viz
