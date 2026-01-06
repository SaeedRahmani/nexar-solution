"""
Scenario Distribution Analysis - VISIBLE ONLY SUBSET

This script analyzes how scenarios are distributed in the dataset and 
how the model performs across different scenario types.

SUBSET: Excludes videos where "Target not visible" is True.

Analyses:
1. Dataset scenario distribution (Level 1 & Level 2)
2. Prediction outcomes by scenario (TP/FP/FN/TN breakdown)
3. Error rate per scenario
4. Scenario representation in errors vs dataset
5. Cross-tabulation heatmaps
6. False negatives analysis (missed accidents)

Usage:
    python prediction_analysis_visible_only/scripts/scenario_distribution_analysis.py

Outputs to prediction_analysis_visible_only/scenario_distribution/:
    - dataset_distribution_level1.png/pdf
    - dataset_distribution_level2.png/pdf
    - prediction_outcomes_level1.png/pdf
    - prediction_outcomes_level2.png/pdf
    - error_rate_by_scenario.png/pdf
    - scenario_representation_comparison.png/pdf
    - scenario_heatmap_level1.png/pdf
    - scenario_heatmap_level2.png/pdf
    - false_negatives_level1.png/pdf
    - false_negatives_level2.png/pdf
    - scenario_statistics.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
BASE_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "video_predictions_ALL/aggregated_predictions_any.csv")
FILTERED_LABELS_PATH = os.path.join(BASE_DIR, "filtered_labels/final_matched_visible_only.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "scenario_distribution")

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'tp': '#2E86AB',
    'tn': '#A23B72',
    'fp': '#F18F01',
    'fn': '#E63946',
    'correct': '#4CAF50',
    'wrong': '#E63946',
    'primary': '#2E86AB',
    'secondary': '#F4A261',
}


def load_data():
    """Load all required data."""
    print("Loading data...")
    
    # Load predictions
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions['video_id'] = predictions['video_id'].astype(str).str.zfill(5)
    print(f"  Predictions: {len(predictions)} videos")
    
    # Load filtered scenario labels (visible only)
    labels = pd.read_csv(FILTERED_LABELS_PATH)
    labels['video_id'] = labels['Video Name'].str.replace('.mp4', '', regex=False)
    labels = labels.rename(columns={
        'Scenario Level 1': 'scenario_level_1',
        'Scenario Level 2': 'scenario_level_2',
    })
    print(f"  Scenario labels (visible only): {len(labels)} videos")
    
    # Merge predictions with scenario labels
    merged = predictions.merge(
        labels[['video_id', 'scenario_level_1', 'scenario_level_2']],
        on='video_id',
        how='inner'
    )
    print(f"  Merged (predictions + scenarios): {len(merged)} videos")
    
    # Add prediction outcome
    merged['outcome'] = 'Unknown'
    merged.loc[(merged['pred'] == 1) & (merged['true_label'] == 1), 'outcome'] = 'TP'
    merged.loc[(merged['pred'] == 0) & (merged['true_label'] == 0), 'outcome'] = 'TN'
    merged.loc[(merged['pred'] == 1) & (merged['true_label'] == 0), 'outcome'] = 'FP'
    merged.loc[(merged['pred'] == 0) & (merged['true_label'] == 1), 'outcome'] = 'FN'
    
    # Add correct/wrong flag
    merged['is_correct'] = merged['pred'] == merged['true_label']
    
    return merged, labels


def plot_dataset_distribution(df, level_col, title_suffix, output_name, output_dir):
    """Plot scenario distribution in the dataset."""
    
    counts = df[level_col].value_counts().sort_values(ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(counts) * 0.4)))
    fig.suptitle(f'Scenario Distribution - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Bar chart
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    bars = ax1.barh(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(counts)))
    ax1.set_yticklabels(counts.index, fontsize=10)
    ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Count per Scenario', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        ax1.text(count + max(counts) * 0.01, i, f'{count}', va='center', fontsize=9)
    
    # Pie chart
    ax2 = axes[1]
    percentages = counts / counts.sum() * 100
    labels = [f'{name}\n({pct:.1f}%)' if pct > 3 else '' 
              for name, pct in zip(counts.index, percentages)]
    wedges, texts = ax2.pie(counts.values, labels=labels, colors=colors,
                            startangle=90, counterclock=False)
    ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return counts


def plot_prediction_outcomes_by_scenario(df, level_col, title_suffix, output_name, output_dir):
    """Plot prediction outcomes (TP/TN/FP/FN) by scenario."""
    
    outcomes = ['TP', 'TN', 'FP', 'FN']
    crosstab = pd.crosstab(df[level_col], df['outcome'])
    
    for outcome in outcomes:
        if outcome not in crosstab.columns:
            crosstab[outcome] = 0
    crosstab = crosstab[outcomes]
    
    crosstab['total'] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values('total', ascending=True)
    crosstab = crosstab.drop('total', axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(crosstab) * 0.5)))
    fig.suptitle(f'Prediction Outcomes by Scenario - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Stacked bar chart (counts)
    ax1 = axes[0]
    bottoms = np.zeros(len(crosstab))
    
    for outcome in outcomes:
        color = COLORS[outcome.lower()]
        ax1.barh(range(len(crosstab)), crosstab[outcome].values, left=bottoms,
                 label=outcome, color=color, edgecolor='white', linewidth=0.5)
        bottoms += crosstab[outcome].values
    
    ax1.set_yticks(range(len(crosstab)))
    ax1.set_yticklabels(crosstab.index, fontsize=10)
    ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Outcome Counts', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Stacked bar chart (percentages)
    ax2 = axes[1]
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    bottoms = np.zeros(len(crosstab_pct))
    
    for outcome in outcomes:
        color = COLORS[outcome.lower()]
        ax2.barh(range(len(crosstab_pct)), crosstab_pct[outcome].values, left=bottoms,
                 label=outcome, color=color, edgecolor='white', linewidth=0.5)
        bottoms += crosstab_pct[outcome].values
    
    ax2.set_yticks(range(len(crosstab_pct)))
    ax2.set_yticklabels(crosstab_pct.index, fontsize=10)
    ax2.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Outcome Percentages', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return crosstab


def plot_error_rate_by_scenario(df, level_col, title_suffix, output_name, output_dir):
    """Plot error rate (% wrong) by scenario."""
    
    error_stats = df.groupby(level_col).agg({
        'is_correct': ['sum', 'count']
    }).reset_index()
    error_stats.columns = [level_col, 'correct_count', 'total_count']
    error_stats['error_count'] = error_stats['total_count'] - error_stats['correct_count']
    error_stats['error_rate'] = (error_stats['error_count'] / error_stats['total_count']) * 100
    error_stats['accuracy'] = (error_stats['correct_count'] / error_stats['total_count']) * 100
    
    error_stats = error_stats.sort_values('error_rate', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(error_stats) * 0.5)))
    fig.suptitle(f'Error Rate by Scenario - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Error rate bar chart
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(error_stats['error_rate'] / max(error_stats['error_rate'].max(), 1))
    bars = ax1.barh(range(len(error_stats)), error_stats['error_rate'].values, 
                    color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(error_stats)))
    ax1.set_yticklabels(error_stats[level_col].values, fontsize=10)
    ax1.set_xlabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Error Rate per Scenario\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, (bar, rate, count) in enumerate(zip(bars, error_stats['error_rate'], error_stats['total_count'])):
        ax1.text(rate + 1, i, f'{rate:.1f}% (n={count})', va='center', fontsize=9)
    
    # Error count vs sample size scatter
    ax2 = axes[1]
    scatter = ax2.scatter(error_stats['total_count'], error_stats['error_rate'],
                          c=error_stats['error_rate'], cmap='RdYlGn_r',
                          s=150, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add all scenario labels with adjustText to prevent overlap
    from adjustText import adjust_text
    texts = []
    for _, row in error_stats.iterrows():
        label = row[level_col][:18] + '...' if len(str(row[level_col])) > 18 else row[level_col]
        # Add small offset so text doesn't start on top of point
        txt = ax2.text(row['total_count'] + 2, row['error_rate'] + 0.3, label, fontsize=12)
        texts.append(txt)
    
    # Adjust text positions to avoid overlap, with connecting lines
    adjust_text(texts, ax=ax2,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                expand_points=(1.5, 1.5),
                force_text=(0.5, 0.5))
    
    ax2.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Rate vs Sample Size', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.colorbar(scatter, ax=ax2, label='Error Rate (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return error_stats


def plot_scenario_representation_comparison(df, level_col, title_suffix, output_name, output_dir):
    """Compare scenario representation: dataset vs errors."""
    
    dataset_dist = df[level_col].value_counts(normalize=True) * 100
    wrong_df = df[~df['is_correct']]
    wrong_dist = wrong_df[level_col].value_counts(normalize=True) * 100 if len(wrong_df) > 0 else pd.Series()
    
    comparison = pd.DataFrame({
        'Dataset %': dataset_dist,
        'Errors %': wrong_dist
    }).fillna(0)
    comparison['Difference'] = comparison['Errors %'] - comparison['Dataset %']
    comparison = comparison.sort_values('Difference', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(comparison) * 0.5)))
    fig.suptitle(f'Scenario Representation: Dataset vs Errors - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Side-by-side comparison
    ax1 = axes[0]
    x = np.arange(len(comparison))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, comparison['Dataset %'], width, label='In Dataset', 
                     color=COLORS['primary'], edgecolor='white')
    bars2 = ax1.barh(x + width/2, comparison['Errors %'], width, label='In Errors', 
                     color=COLORS['wrong'], edgecolor='white')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(comparison.index, fontsize=10)
    ax1.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Dataset vs Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Difference plot
    ax2 = axes[1]
    colors = ['#E63946' if d > 0 else '#4CAF50' for d in comparison['Difference']]
    bars = ax2.barh(range(len(comparison)), comparison['Difference'], color=colors,
                    edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(comparison)))
    ax2.set_yticklabels(comparison.index, fontsize=10)
    ax2.set_xlabel('Difference (Errors % - Dataset %)', fontsize=12, fontweight='bold')
    ax2.set_title('Over/Under-representation in Errors\n(Red = over, Green = under)', 
                  fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for i, (bar, diff) in enumerate(zip(bars, comparison['Difference'])):
        sign = '+' if diff > 0 else ''
        ax2.text(diff + 0.5 if diff >= 0 else diff - 0.5, i, 
                 f'{sign}{diff:.1f}%', va='center', fontsize=9,
                 ha='left' if diff >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return comparison


def plot_scenario_heatmap(df, level_col, title_suffix, output_name, output_dir):
    """Create heatmap of scenario × prediction outcome."""
    
    crosstab = pd.crosstab(df[level_col], df['outcome'], normalize='index') * 100
    
    for outcome in ['TP', 'TN', 'FP', 'FN']:
        if outcome not in crosstab.columns:
            crosstab[outcome] = 0
    crosstab = crosstab[['TP', 'TN', 'FP', 'FN']]
    
    crosstab['error_rate'] = crosstab['FP'] + crosstab['FN']
    crosstab = crosstab.sort_values('error_rate', ascending=False)
    crosstab = crosstab.drop('error_rate', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(crosstab) * 0.4)))
    
    im = ax.imshow(crosstab.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['TP\n(Correct)', 'TN\n(Correct)', 'FP\n(Error)', 'FN\n(Error)'], fontsize=11)
    ax.set_yticks(range(len(crosstab)))
    ax.set_yticklabels(crosstab.index, fontsize=10)
    
    for i in range(len(crosstab)):
        for j in range(4):
            val = crosstab.iloc[i, j]
            color = 'white' if val > 50 or val < 20 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color=color)
    
    ax.set_title(f'Prediction Outcome by Scenario (%) - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=14, fontweight='bold', pad=10)
    
    plt.colorbar(im, ax=ax, label='Percentage (%)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return crosstab


def plot_fn_analysis(df, level_col, title_suffix, output_name, output_dir):
    """Special analysis for False Negatives (missed accidents)."""
    
    fn_df = df[df['outcome'] == 'FN']
    total_fn = len(fn_df)
    
    if total_fn == 0:
        print(f"  ⚠️ No false negatives found for {title_suffix}")
        return None
    
    fn_counts = fn_df[level_col].value_counts()
    
    positive_df = df[df['true_label'] == 1]
    miss_rate = fn_df.groupby(level_col).size() / positive_df.groupby(level_col).size() * 100
    miss_rate = miss_rate.fillna(0).sort_values(ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(fn_counts) * 0.5)))
    fig.suptitle(f'⚠️ FALSE NEGATIVES (Missed Accidents) - {title_suffix}\n[VISIBLE ONLY SUBSET]', 
                 fontsize=16, fontweight='bold', color=COLORS['fn'], y=1.02)
    
    # FN count per scenario
    ax1 = axes[0]
    fn_counts_sorted = fn_counts.sort_values(ascending=True)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(fn_counts_sorted)))
    bars = ax1.barh(range(len(fn_counts_sorted)), fn_counts_sorted.values, 
                    color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(fn_counts_sorted)))
    ax1.set_yticklabels(fn_counts_sorted.index, fontsize=10)
    ax1.set_xlabel('Count of Missed Accidents', fontsize=12, fontweight='bold')
    ax1.set_title(f'Missed Accidents by Scenario\n(Total FN: {total_fn})', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, (bar, count) in enumerate(zip(bars, fn_counts_sorted.values)):
        ax1.text(count + 0.1, i, f'{count}', va='center', fontsize=9)
    
    # Miss rate
    ax2 = axes[1]
    colors2 = plt.cm.Reds(miss_rate.values / max(miss_rate.values) * 0.6 + 0.3) if miss_rate.max() > 0 else ['red']
    bars2 = ax2.barh(range(len(miss_rate)), miss_rate.values, 
                     color=colors2, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(miss_rate)))
    ax2.set_yticklabels(miss_rate.index, fontsize=10)
    ax2.set_xlabel('Miss Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('% of Positive Videos Missed\n(Higher = More Dangerous)', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for i, (bar, rate) in enumerate(zip(bars2, miss_rate.values)):
        ax2.text(rate + 0.5, i, f'{rate:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return fn_counts, miss_rate


def save_statistics(df, output_dir):
    """Save detailed statistics to text file."""
    
    with open(os.path.join(output_dir, 'scenario_statistics.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SCENARIO DISTRIBUTION ANALYSIS - VISIBLE ONLY SUBSET\n")
        f.write("(Excluding videos where 'Target not visible' is True)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total videos analyzed: {len(df)}\n")
        f.write(f"Correct predictions: {df['is_correct'].sum()} ({df['is_correct'].mean()*100:.2f}%)\n")
        f.write(f"Wrong predictions: {(~df['is_correct']).sum()} ({(~df['is_correct']).mean()*100:.2f}%)\n\n")
        
        f.write("PREDICTION OUTCOME BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for outcome in ['TP', 'TN', 'FP', 'FN']:
            count = (df['outcome'] == outcome).sum()
            pct = count / len(df) * 100
            f.write(f"  {outcome}: {count} ({pct:.2f}%)\n")
        f.write("\n")
        
        # Level 1 statistics
        f.write("=" * 80 + "\n")
        f.write("LEVEL 1 SCENARIO STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        level1_stats = df.groupby('scenario_level_1').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'outcome': lambda x: (x == 'FN').sum()
        }).reset_index()
        level1_stats.columns = ['Scenario', 'Total', 'Correct', 'Accuracy', 'FN_Count']
        level1_stats['Error_Rate'] = (1 - level1_stats['Accuracy']) * 100
        level1_stats['Accuracy'] = level1_stats['Accuracy'] * 100
        level1_stats = level1_stats.sort_values('Error_Rate', ascending=False)
        
        f.write(f"{'Scenario':<30} {'Total':>8} {'Correct':>8} {'Acc%':>8} {'Err%':>8} {'FN':>6}\n")
        f.write("-" * 80 + "\n")
        for _, row in level1_stats.iterrows():
            f.write(f"{str(row['Scenario'])[:30]:<30} {row['Total']:>8.0f} {row['Correct']:>8.0f} "
                    f"{row['Accuracy']:>7.1f}% {row['Error_Rate']:>7.1f}% {row['FN_Count']:>6.0f}\n")
        f.write("\n")
        
        # Level 2 statistics
        f.write("=" * 80 + "\n")
        f.write("LEVEL 2 SCENARIO STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        level2_stats = df.groupby('scenario_level_2').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'outcome': lambda x: (x == 'FN').sum()
        }).reset_index()
        level2_stats.columns = ['Scenario', 'Total', 'Correct', 'Accuracy', 'FN_Count']
        level2_stats['Error_Rate'] = (1 - level2_stats['Accuracy']) * 100
        level2_stats['Accuracy'] = level2_stats['Accuracy'] * 100
        level2_stats = level2_stats.sort_values('Error_Rate', ascending=False)
        
        f.write(f"{'Scenario':<40} {'Total':>8} {'Correct':>8} {'Acc%':>8} {'Err%':>8} {'FN':>6}\n")
        f.write("-" * 90 + "\n")
        for _, row in level2_stats.iterrows():
            f.write(f"{str(row['Scenario'])[:40]:<40} {row['Total']:>8.0f} {row['Correct']:>8.0f} "
                    f"{row['Accuracy']:>7.1f}% {row['Error_Rate']:>7.1f}% {row['FN_Count']:>6.0f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✅ Saved: scenario_statistics.txt")


def main():
    print("=" * 80)
    print("SCENARIO DISTRIBUTION ANALYSIS - VISIBLE ONLY SUBSET")
    print("=" * 80)
    
    # Load data
    df, labels = load_data()
    
    print(f"\n{'='*80}")
    print("GENERATING LEVEL 1 ANALYSES")
    print(f"{'='*80}\n")
    
    plot_dataset_distribution(df, 'scenario_level_1', 'Level 1', 
                              'dataset_distribution_level1', OUTPUT_DIR)
    plot_prediction_outcomes_by_scenario(df, 'scenario_level_1', 'Level 1', 
                                         'prediction_outcomes_level1', OUTPUT_DIR)
    plot_error_rate_by_scenario(df, 'scenario_level_1', 'Level 1', 
                                'error_rate_level1', OUTPUT_DIR)
    plot_scenario_representation_comparison(df, 'scenario_level_1', 'Level 1', 
                                            'representation_comparison_level1', OUTPUT_DIR)
    plot_scenario_heatmap(df, 'scenario_level_1', 'Level 1', 
                          'scenario_heatmap_level1', OUTPUT_DIR)
    plot_fn_analysis(df, 'scenario_level_1', 'Level 1', 
                     'false_negatives_level1', OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("GENERATING LEVEL 2 ANALYSES")
    print(f"{'='*80}\n")
    
    plot_dataset_distribution(df, 'scenario_level_2', 'Level 2', 
                              'dataset_distribution_level2', OUTPUT_DIR)
    plot_prediction_outcomes_by_scenario(df, 'scenario_level_2', 'Level 2', 
                                         'prediction_outcomes_level2', OUTPUT_DIR)
    plot_error_rate_by_scenario(df, 'scenario_level_2', 'Level 2', 
                                'error_rate_level2', OUTPUT_DIR)
    plot_scenario_representation_comparison(df, 'scenario_level_2', 'Level 2', 
                                            'representation_comparison_level2', OUTPUT_DIR)
    plot_scenario_heatmap(df, 'scenario_level_2', 'Level 2', 
                          'scenario_heatmap_level2', OUTPUT_DIR)
    plot_fn_analysis(df, 'scenario_level_2', 'Level 2', 
                     'false_negatives_level2', OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("SAVING STATISTICS")
    print(f"{'='*80}\n")
    
    save_statistics(df, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
