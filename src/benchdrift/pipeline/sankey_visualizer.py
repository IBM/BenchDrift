"""
Sankey Diagram Visualizer for Variation Pipeline
Generates hierarchical Sankey diagrams showing transformation pipeline flow
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class Visualizer:
    """Creates interactive Sankey diagrams for variation pipeline analysis"""

    def __init__(self, df):
        """
        Initialize with dataframe containing variation data

        Args:
            df: DataFrame with columns 'is_variant', 'transformation_type', 'debugging_capability'
        """
        self.df = df
        self.variants_df = df[df['is_variant']].copy()

    def create_result_summary(self):
        """Create paper-ready result summary with enhanced styling"""
        import matplotlib.patches as mpatches

        # Paper-ready styling with Arial font
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 14,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

        problems_count = self.df['problem_id'].nunique()
        variants_count = self.df['is_variant'].sum()

        # Create 2-panel layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Variation Categories - SMALLER PIE CHART with LARGER FONTS
        if 'transformation_type' in self.df.columns:
            trans_types = self.df[self.df['is_variant']]['transformation_type'].fillna('unknown')

            # Count by prefix-based categories
            category_counts = {}
            category_details = {}

            # Generic transformations
            generic_types = [t for t in trans_types if t.startswith('generic_')]
            if generic_types:
                category_counts['Generic'] = len(generic_types)
                subcats = set()
                for t in generic_types[:20]:
                    clean_type = t.replace('generic_', '').replace('_', ' ').title()
                    subcats.add(clean_type)
                category_details['Generic'] = list(subcats)[:2]

            # Persona transformations
            persona_types = [t for t in trans_types if t.startswith('persona_')]
            if persona_types:
                category_counts['Personas'] = len(persona_types)
                subcats = set()
                for t in persona_types[:20]:
                    clean_type = t.replace('persona_', '').replace('_persona', '').replace('_', ' ').title()
                    subcats.add(clean_type)
                category_details['Personas'] = list(subcats)[:2]

            # Combination transformations
            combo_types = [t for t in trans_types if t.startswith('bucket_combination_')]
            if combo_types:
                category_counts['Combinations'] = len(combo_types)
                subcats = set()
                for t in combo_types[:20]:
                    clean_type = t.replace('bucket_combination_', '').replace('_', ' ').title()
                    subcats.add(clean_type)
                category_details['Combinations'] = list(subcats)[:2]

            # Create pie chart
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            colors = ['#FAC858', '#73C0DE', '#3BA272'][:len(categories)]

            # Create labels with subcategories on separate lines
            labels = []
            for category in categories:
                subcats = category_details.get(category, [])
                if subcats:
                    subcat_text = ',\n'.join(subcats[:2])
                    labels.append(f'{category}\n({subcat_text})\n[{category_counts[category]}]')
                else:
                    labels.append(f'{category}\n[{category_counts[category]}]')

            # Smaller pie with larger text
            wedges, texts, autotexts = ax1.pie(counts, labels=labels, colors=colors,
                                              autopct='%1.0f%%', startangle=90,
                                              radius=0.7,  # Smaller pie
                                              textprops={'fontsize': 12},
                                              pctdistance=0.85)

            ax1.set_title('Variation Categories', fontweight='bold', fontsize=16)

        else:
            ax1.text(0.5, 0.5, 'No variation\ntype data', ha='center', va='center', fontsize=14)
            ax1.set_title('Variation Categories', fontweight='bold', fontsize=16)

        # 2. Model Performance Impact with CENTERED LEGEND
        if 'has_drift' in self.df.columns:
            # Calculate baseline performance
            baseline_correct = (~self.df['has_drift'][self.df['is_baseline']]).sum() if self.df['is_baseline'].sum() > 0 else 0
            baseline_total = self.df['is_baseline'].sum()
            baseline_acc = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0

            # Calculate variation performance segments
            variants_df = self.df[self.df['is_variant']]
            stable_count = (~variants_df['has_drift']).sum()
            improvement_count = variants_df['has_improvement'].sum() if 'has_improvement' in variants_df.columns else 0
            negative_drift_count = variants_df['has_drift'].sum() - improvement_count

            stable_pct = (stable_count / len(variants_df)) * 100
            improvement_pct = (improvement_count / len(variants_df)) * 100
            negative_pct = (negative_drift_count / len(variants_df)) * 100

            # Create baseline bar
            bars = ax2.bar(['Baseline'], [baseline_acc], color='#4472C4', width=0.5)

            # Stacked segments for variations
            ax2.bar(['Variations'], [stable_pct], color='#70AD47', width=0.5, label='Stable')
            ax2.bar(['Variations'], [improvement_pct], bottom=stable_pct, color='#2E8B57', width=0.5, label='Improved')
            ax2.bar(['Variations'], [negative_pct], bottom=stable_pct+improvement_pct, color='#E74C3C', width=0.5, label='Degraded')

            ax2.set_ylabel('Performance (%)', fontsize=14)
            ax2.set_title('Performance Impact', fontweight='bold', fontsize=16)
            ax2.set_ylim(0, 105)

            # Add baseline percentage label
            ax2.text(0, baseline_acc + 2, f'{baseline_acc:.0f}%', ha='center', fontweight='bold', fontsize=12)

            # Add variation segment labels (only if large enough)
            if stable_pct > 8:
                ax2.text(1, stable_pct/2, f'{stable_pct:.0f}%', ha='center', va='center',
                        fontweight='bold', color='white', fontsize=11)
            if improvement_pct > 8:
                ax2.text(1, stable_pct + improvement_pct/2, f'{improvement_pct:.0f}%',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=11)
            if negative_pct > 8:
                ax2.text(1, stable_pct + improvement_pct + negative_pct/2, f'{negative_pct:.0f}%',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=11)

            # CENTERED LEGEND
            ax2.legend(loc='center', fontsize=12, frameon=True, fancybox=True, shadow=True)

        else:
            ax2.text(0.5, 0.5, 'Performance analysis\nrequires drift evaluation',
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Performance Impact', fontweight='bold', fontsize=16)

        plt.tight_layout()
        plt.show()

        # Enhanced Pipeline Insights with COLORED Important Details
        if 'has_drift' in self.df.columns and 'transformation_type' in self.df.columns:
            drift_rate = self.df[self.df['is_variant']]['has_drift'].sum() / self.df['is_variant'].sum() * 100

            variants_df = self.df[self.df['is_variant']].copy()
            trans_clean = [t.replace('generic_', '').replace('_', ' ').title() for t in variants_df['transformation_type'].fillna('unknown')]
            variants_df['trans_clean'] = trans_clean

            drift_by_type = variants_df.groupby('trans_clean')['has_drift'].mean().sort_values(ascending=False)
            most_problematic = drift_by_type.index[0] if len(drift_by_type) > 0 else 'Unknown'
            problematic_rate = drift_by_type.iloc[0] * 100 if len(drift_by_type) > 0 else 0

            # Drift breakdown by variation category
            variants_df['variation_category'] = 'Other'
            variants_df.loc[variants_df['transformation_type'].str.startswith('generic_', na=False), 'variation_category'] = 'Generic'
            variants_df.loc[variants_df['transformation_type'].str.startswith('persona_', na=False), 'variation_category'] = 'Personas'
            variants_df.loc[variants_df['transformation_type'].str.startswith('bucket_combination_', na=False), 'variation_category'] = 'Combinations'

            # Negative drift breakdown
            neg_drift_df = variants_df[variants_df['has_drift'] & ~variants_df.get('has_improvement', False)]
            neg_drift_breakdown = neg_drift_df['variation_category'].value_counts(normalize=True) * 100 if len(neg_drift_df) > 0 else pd.Series()

            # Positive drift breakdown (if exists)
            pos_drift_df = variants_df[variants_df.get('has_improvement', False)] if 'has_improvement' in variants_df.columns else pd.DataFrame()
            pos_drift_breakdown = pos_drift_df['variation_category'].value_counts(normalize=True) * 100 if len(pos_drift_df) > 0 else pd.Series()

        # Print pipeline insights as clean text output
        print("\n" + "="*60)
        print("📋 PIPELINE INSIGHTS")
        print("="*60)
        print(f"✓ Generated {variants_count} variations from {problems_count} problem(s)")

        # Drift rate with light background highlighting
        if drift_rate > 15:
            drift_status = f"\033[101m{drift_rate:.0f}%\033[0m (HIGH)"  # Light red background
        elif drift_rate > 5:
            drift_status = f"\033[103m{drift_rate:.0f}%\033[0m (MEDIUM)"  # Lemon yellow background
        else:
            drift_status = f"\033[102m{drift_rate:.0f}%\033[0m (LOW)"  # Light green background

        print(f"✓ Model drift rate: {drift_status}")
        print(f"✓ Most challenging type: \033[103m{most_problematic}\033[0m ({problematic_rate:.0f}% drift rate)")

        # Drift composition
        if len(neg_drift_breakdown) > 0:
            neg_items = list(neg_drift_breakdown.items())[:3]
            neg_text = ", ".join([f"{cat} {pct:.0f}%" for cat, pct in neg_items])
            print(f"✓ Negative drift breakdown: \033[105m{neg_text}\033[0m")

        if len(pos_drift_breakdown) > 0:
            pos_items = list(pos_drift_breakdown.items())[:3]
            pos_text = ", ".join([f"{cat} {pct:.0f}%" for cat, pct in pos_items])
            print(f"✓ Positive drift breakdown: \033[102m{pos_text}\033[0m")

        print("✓ Pipeline successfully identified model robustness issues")
        print("="*60)

    def display_sankey(self, **kwargs):
        """Display the Sankey diagram with debugging capabilities"""
        # Enhanced Hierarchical Sankey: With Debugging Capabilities (Level 3)
        try:
            import plotly.graph_objects as go
            from plotly.offline import iplot, init_notebook_mode

            df = self.df

            if 'transformation_type' in df.columns and 'debugging_capability' in df.columns:
                trans_types = df[df['is_variant']]['transformation_type'].fillna('unknown')
                debug_caps = df[df['is_variant']]['debugging_capability'].fillna('unknown')

                # STEP 1: Build Level 1 categories
                level1_counts = {'Generic': 0, 'Candidates': 0, 'Personas': 0}
                level1_items = {'Generic': [], 'Candidates': [], 'Personas': []}

                variants_df = df[df['is_variant']].copy()

                for i, trans_type in enumerate(trans_types):
                    debug_cap = debug_caps.iloc[i]

                    if trans_type.startswith('generic_'):
                        level1_counts['Generic'] += 1
                        level1_items['Generic'].append((trans_type, debug_cap))
                    elif trans_type.startswith('persona_'):
                        level1_counts['Personas'] += 1
                        level1_items['Personas'].append((trans_type, debug_cap))
                    else:
                        level1_counts['Candidates'] += 1
                        level1_items['Candidates'].append((trans_type, debug_cap))

                print(f"Level 1 distribution:")
                for cat, count in level1_counts.items():
                    print(f"  {cat}: {count} variations")

                # STEP 2: Build Level 2 subcategories (transformation types)
                level2_subcats = {'Generic': {}, 'Candidates': {}, 'Personas': {}}
                level2_to_debug = {}  # Track which debug caps go with which level2 subcats

                for category, items in level1_items.items():
                    for trans_type, debug_cap in items:
                        if category == 'Generic' and trans_type.startswith('generic_'):
                            subcat = trans_type[8:].replace('_', ' ').title()
                        elif category == 'Personas' and trans_type.startswith('persona_'):
                            subcat = trans_type[8:].replace('_', ' ').title()
                        elif category == 'Candidates':
                            if trans_type.startswith('bucket_combination_'):
                                subcat = trans_type[19:].replace('_', ' ').title() + ' Combination'
                            else:
                                subcat = trans_type.replace('_', ' ').title()
                        else:
                            subcat = 'Other'

                        level2_subcats[category][subcat] = level2_subcats[category].get(subcat, 0) + 1

                        # Track debug capabilities for this subcat
                        level2_key = f"{category}_{subcat}"
                        if level2_key not in level2_to_debug:
                            level2_to_debug[level2_key] = {}
                        level2_to_debug[level2_key][debug_cap] = level2_to_debug[level2_key].get(debug_cap, 0) + 1

                # STEP 3: Build Level 3 debugging capabilities
                all_debug_caps = set()
                for debug_dict in level2_to_debug.values():
                    all_debug_caps.update(debug_dict.keys())
                all_debug_caps.discard('unknown')  # Remove unknown
                all_debug_caps = list(all_debug_caps)

                print(f"\nLevel 3 debugging capabilities found: {len(all_debug_caps)}")

                # STEP 4: Build Sankey nodes
                nodes = ['Original Problems']  # Level 0
                node_colors = ['#4472C4']

                # Level 1 nodes
                level1_start = len(nodes)
                for category, count in level1_counts.items():
                    if count > 0:
                        nodes.append(category)
                        if category == 'Generic':
                            node_colors.append('#FAC858')
                        elif category == 'Candidates':
                            node_colors.append('#73C0DE')
                        else:  # Personas
                            node_colors.append('#3BA272')

                # Level 2 nodes (transformation subcategories)
                level2_start = len(nodes)
                level2_mapping = {}  # Map level2 keys to node indices

                for category in ['Generic', 'Candidates', 'Personas']:
                    if level1_counts[category] > 0:
                        for subcat, count in level2_subcats[category].items():
                            if count > 0:
                                level2_key = f"{category}_{subcat}"
                                level2_mapping[level2_key] = len(nodes)
                                nodes.append(f"{subcat}")
                                if category == 'Generic':
                                    node_colors.append('#FFD700')
                                elif category == 'Candidates':
                                    node_colors.append('#87CEEB')
                                else:  # Personas
                                    node_colors.append('#90EE90')

                # Level 3 nodes (debugging capabilities) - only top 10 most common
                level3_start = len(nodes)
                debug_cap_counts = {}
                for debug_dict in level2_to_debug.values():
                    for cap, count in debug_dict.items():
                        if cap != 'unknown':
                            debug_cap_counts[cap] = debug_cap_counts.get(cap, 0) + count

                # Get top 10 most common debugging capabilities
                top_debug_caps = sorted(debug_cap_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                selected_debug_caps = [cap for cap, _ in top_debug_caps]

                for debug_cap in selected_debug_caps:
                    clean_cap = debug_cap.replace('_', ' ').title()
                    nodes.append(clean_cap)
                    node_colors.append('#FFB6C1')  # Light pink for debugging capabilities

                print(f"Selected top 10 debugging capabilities: {selected_debug_caps}")

                # STEP 5: Create links
                source_indices = []
                target_indices = []
                values = []
                link_colors = []

                # Level 0 -> Level 1 links
                level1_nodes = []
                level1_idx = level1_start
                for category, count in level1_counts.items():
                    if count > 0:
                        level1_nodes.append(category)
                        source_indices.append(0)  # Original Problems
                        target_indices.append(level1_idx)
                        values.append(count)
                        if category == 'Generic':
                            link_colors.append('rgba(250, 200, 88, 0.4)')
                        elif category == 'Candidates':
                            link_colors.append('rgba(115, 192, 222, 0.4)')
                        else:  # Personas
                            link_colors.append('rgba(58, 178, 114, 0.4)')
                        level1_idx += 1

                # Level 1 -> Level 2 links
                level1_idx = level1_start
                for i, category in enumerate(level1_nodes):
                    for subcat, count in level2_subcats[category].items():
                        if count > 0:
                            level2_key = f"{category}_{subcat}"
                            if level2_key in level2_mapping:
                                source_indices.append(level1_idx)
                                target_indices.append(level2_mapping[level2_key])
                                values.append(count)
                                if category == 'Generic':
                                    link_colors.append('rgba(255, 215, 0, 0.3)')
                                elif category == 'Candidates':
                                    link_colors.append('rgba(135, 206, 235, 0.3)')
                                else:  # Personas
                                    link_colors.append('rgba(144, 238, 144, 0.3)')
                    level1_idx += 1

                # Level 2 -> Level 3 links (to debugging capabilities)
                for level2_key, debug_dict in level2_to_debug.items():
                    if level2_key in level2_mapping:
                        level2_idx = level2_mapping[level2_key]
                        for debug_cap, count in debug_dict.items():
                            if debug_cap in selected_debug_caps:
                                level3_idx = level3_start + selected_debug_caps.index(debug_cap)
                                source_indices.append(level2_idx)
                                target_indices.append(level3_idx)
                                values.append(count)
                                link_colors.append('rgba(255, 182, 193, 0.3)')

                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=12,
                        thickness=15,
                        line=dict(color="black", width=0.3),
                        label=nodes,
                        color=node_colors
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values,
                        color=link_colors
                    )
                )])

                fig.update_layout(
                    title_text="Complete Pipeline: Transformation Methods → Debugging Capabilities",
                    font_size=10,
                    width=1200,
                    height=700
                )

                fig.show()

            else:
                print("Missing required fields: transformation_type and/or debugging_capability")

        except ImportError:
            print("📊 Enhanced Pipeline with Debugging Capabilities")
            print("=" * 60)
            print("Install plotly to see interactive 3-level Sankey diagram")
            print("\nStructure:")
            print("Level 0: Original Problems")
            print("Level 1: Generic | Personas | Candidates")
            print("Level 2: Transformation subcategories")
            print("Level 3: Debugging capabilities (what's being tested)")

        print("\n🎯 Enhanced Sankey shows: HOW variations are created → WHAT capabilities are tested")