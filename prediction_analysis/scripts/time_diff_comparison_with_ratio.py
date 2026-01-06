"""
Time Difference Comparison with Ratio Line

This script creates overlapping histograms comparing time differences between
wrong predictions (false negatives) and correct predictions (true positives),
with a ratio line showing the wrong/correct ratio per bin.

This analysis helps identify if there's a relationship between the timing of
accidents (time_of_event - time_of_alert) and prediction accuracy.

Usage:
    python prediction_analysis/scripts/time_diff_comparison_with_ratio.py

Outputs to prediction_analysis/comparative_analysis/:
    - comparison_overlapping_histograms.png/pdf (with ratio line)
    - comparison_overlapping_histograms_normalized.png/pdf (normalized with ratio)
    - comparison_histograms.png/pdf (side-by-side)
    - comparison_combined_boxplot.png/pdf
    - comparison_scatter_plots.png/pdf
    - comparison_violin_plot.png/pdf
    - correct_predictions_analysis.png/pdf (individual analysis for TP)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
WRONG_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/wrong_predictions")
CORRECT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/correct_predictions")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/comparative_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
colors = {
    'wrong': '#E63946',      # Red for wrong predictions
    'correct': '#2A9D8F',    # Green for correct predictions
    'accent': '#F4A261',
    'dark': '#264653',
    'light': '#E9C46A'
}


def load_data():
    """Load all prediction data."""
    fn_df = pd.read_csv(os.path.join(WRONG_DIR, 'false_negatives.csv'), dtype={'video_id': str})
    fp_df = pd.read_csv(os.path.join(WRONG_DIR, 'false_positives.csv'), dtype={'video_id': str})
    tp_df = pd.read_csv(os.path.join(CORRECT_DIR, 'true_positives.csv'), dtype={'video_id': str})
    tn_df = pd.read_csv(os.path.join(CORRECT_DIR, 'true_negatives.csv'), dtype={'video_id': str})
    
    return fn_df, fp_df, tp_df, tn_df


def print_statistics(false_negatives, true_positives):
    """Print comparison statistics."""
    print("=" * 80)
    print("TIME DIFFERENCE ANALYSIS: WRONG vs CORRECT PREDICTIONS")
    print("=" * 80)
    
    print(f"\n{'='*80}")
    print("WRONG PREDICTIONS - FALSE NEGATIVES")
    print(f"{'='*80}")
    print(f"Total false negatives: {len(false_negatives)}")
    
    if len(false_negatives) > 0:
        print(f"\nTime Difference Statistics (Event Time - Alert Time):")
        print(f"  Mean:   {false_negatives['time_difference'].mean():.3f} seconds")
        print(f"  Std:    {false_negatives['time_difference'].std():.3f} seconds")
        print(f"  Median: {false_negatives['time_difference'].median():.3f} seconds")
        print(f"  Min:    {false_negatives['time_difference'].min():.3f} seconds")
        print(f"  Max:    {false_negatives['time_difference'].max():.3f} seconds")
        print(f"  Q1:     {false_negatives['time_difference'].quantile(0.25):.3f} seconds")
        print(f"  Q3:     {false_negatives['time_difference'].quantile(0.75):.3f} seconds")
    
    print(f"\n{'='*80}")
    print("CORRECT PREDICTIONS - TRUE POSITIVES")
    print(f"{'='*80}")
    print(f"Total true positives: {len(true_positives)}")
    
    if len(true_positives) > 0:
        print(f"\nTime Difference Statistics (Event Time - Alert Time):")
        print(f"  Mean:   {true_positives['time_difference'].mean():.3f} seconds")
        print(f"  Std:    {true_positives['time_difference'].std():.3f} seconds")
        print(f"  Median: {true_positives['time_difference'].median():.3f} seconds")
        print(f"  Min:    {true_positives['time_difference'].min():.3f} seconds")
        print(f"  Max:    {true_positives['time_difference'].max():.3f} seconds")
        print(f"  Q1:     {true_positives['time_difference'].quantile(0.25):.3f} seconds")
        print(f"  Q3:     {true_positives['time_difference'].quantile(0.75):.3f} seconds")
    
    # Statistical comparison
    if len(false_negatives) > 0 and len(true_positives) > 0:
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*80}")
        
        # Mann-Whitney U test (non-parametric)
        stat, pvalue = stats.mannwhitneyu(
            false_negatives['time_difference'], 
            true_positives['time_difference'],
            alternative='two-sided'
        )
        print(f"\nMann-Whitney U Test (non-parametric):")
        print(f"  U-statistic: {stat:.4f}")
        print(f"  P-value: {pvalue:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if pvalue < 0.05 else 'No'}")
        
        # Effect size (Cohen's d)
        mean_diff = false_negatives['time_difference'].mean() - true_positives['time_difference'].mean()
        pooled_std = np.sqrt(
            (false_negatives['time_difference'].std()**2 + true_positives['time_difference'].std()**2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: ", end="")
        if abs(cohens_d) < 0.2:
            print("Negligible")
        elif abs(cohens_d) < 0.5:
            print("Small")
        elif abs(cohens_d) < 0.8:
            print("Medium")
        else:
            print("Large")


def plot_overlapping_histograms_with_ratio(false_negatives, true_positives, output_dir):
    """Create overlapping histograms with ratio line."""
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Define common bins for both histograms
    all_time_diffs = np.concatenate([false_negatives['time_difference'].values,
                                     true_positives['time_difference'].values])
    bin_edges = np.linspace(all_time_diffs.min(), all_time_diffs.max(), 16)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histograms with same bins
    n_wrong, _, _ = ax1.hist(false_negatives['time_difference'], bins=bin_edges,
                              color=colors['wrong'], edgecolor='white', linewidth=1.2, alpha=0.6,
                              label=f'Wrong (n={len(false_negatives)}, μ={false_negatives["time_difference"].mean():.2f}s)')
    n_correct, _, _ = ax1.hist(true_positives['time_difference'], bins=bin_edges,
                               color=colors['correct'], edgecolor='white', linewidth=1.2, alpha=0.6,
                               label=f'Correct (n={len(true_positives)}, μ={true_positives["time_difference"].mean():.2f}s)')
    
    # Add mean lines
    ax1.axvline(false_negatives['time_difference'].mean(), color=colors['wrong'],
                linestyle='--', linewidth=2.5, alpha=0.9)
    ax1.axvline(true_positives['time_difference'].mean(), color=colors['correct'],
                linestyle='--', linewidth=2.5, alpha=0.9)
    
    ax1.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Time Difference Distribution: Wrong vs Correct Predictions\n(Overlapping with Error Ratio)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Create secondary y-axis for ratio
    ax2 = ax1.twinx()
    
    # Calculate ratio (wrong/correct) for each bin, handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(n_correct > 0, n_wrong / n_correct, np.nan)
    
    # Plot ratio line (only where we have valid ratios)
    valid_mask = ~np.isnan(ratio)
    ax2.plot(bin_centers[valid_mask], ratio[valid_mask],
             color=colors['dark'], linewidth=2.5, marker='o', markersize=8,
             label='Wrong/Correct Ratio', zorder=10)
    
    # Add markers for bins with no correct predictions (ratio undefined)
    if np.any(~valid_mask & (n_wrong > 0)):
        undefined_bins = bin_centers[~valid_mask & (n_wrong > 0)]
        ax2.scatter(undefined_bins, [ax2.get_ylim()[1] * 0.9] * len(undefined_bins),
                    marker='^', s=100, color=colors['dark'],
                    label='Undefined (no correct)', zorder=10)
    
    ax2.set_ylabel('Wrong/Correct Ratio', fontsize=12, fontweight='bold', color=colors['dark'])
    ax2.tick_params(axis='y', labelcolor=colors['dark'])
    ax2.spines['top'].set_visible(False)
    ax2.legend(fontsize=10, loc='upper left')
    
    # Add horizontal line at ratio = 1 for reference
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_overlapping_histograms.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_overlapping_histograms.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_overlapping_histograms.png/pdf")
    
    return bin_edges, bin_centers, n_wrong, n_correct, ratio, valid_mask


def plot_normalized_overlapping_histograms(false_negatives, true_positives, bin_edges, bin_centers, 
                                           ratio, valid_mask, output_dir):
    """Create normalized overlapping histograms with ratio line."""
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot normalized histograms (density)
    n_wrong_norm, _, _ = ax1.hist(false_negatives['time_difference'], bins=bin_edges,
                                   color=colors['wrong'], edgecolor='white', linewidth=1.2, alpha=0.6,
                                   density=True,
                                   label=f'Wrong (n={len(false_negatives)}, μ={false_negatives["time_difference"].mean():.2f}s)')
    n_correct_norm, _, _ = ax1.hist(true_positives['time_difference'], bins=bin_edges,
                                     color=colors['correct'], edgecolor='white', linewidth=1.2, alpha=0.6,
                                     density=True,
                                     label=f'Correct (n={len(true_positives)}, μ={true_positives["time_difference"].mean():.2f}s)')
    
    # Add mean lines
    ax1.axvline(false_negatives['time_difference'].mean(), color=colors['wrong'],
                linestyle='--', linewidth=2.5, alpha=0.9)
    ax1.axvline(true_positives['time_difference'].mean(), color=colors['correct'],
                linestyle='--', linewidth=2.5, alpha=0.9)
    
    ax1.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density (Normalized)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Difference Distribution (Normalized): Wrong vs Correct Predictions\n(with Error Ratio based on counts)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Create secondary y-axis for ratio (using original counts)
    ax2 = ax1.twinx()
    
    # Plot ratio line
    ax2.plot(bin_centers[valid_mask], ratio[valid_mask],
             color=colors['dark'], linewidth=2.5, marker='o', markersize=8,
             label='Wrong/Correct Ratio', zorder=10)
    
    ax2.set_ylabel('Wrong/Correct Ratio', fontsize=12, fontweight='bold', color=colors['dark'])
    ax2.tick_params(axis='y', labelcolor=colors['dark'])
    ax2.spines['top'].set_visible(False)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_overlapping_histograms_normalized.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_overlapping_histograms_normalized.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_overlapping_histograms_normalized.png/pdf")


def plot_side_by_side_histograms(false_negatives, true_positives, output_dir):
    """Create side-by-side histograms."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Time Difference Distribution Comparison: Wrong vs Correct Predictions',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Wrong predictions histogram
    axes[0].hist(false_negatives['time_difference'], bins=10,
                 color=colors['wrong'], edgecolor='white', linewidth=1.2, alpha=0.8)
    axes[0].axvline(false_negatives['time_difference'].mean(), color=colors['dark'],
                    linestyle='--', linewidth=2, label=f'Mean: {false_negatives["time_difference"].mean():.2f}s')
    axes[0].set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0].set_title(f'WRONG Predictions (n={len(false_negatives)})\n(False Negatives)',
                      fontsize=12, fontweight='bold', color=colors['wrong'])
    axes[0].legend(fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Correct predictions histogram
    axes[1].hist(true_positives['time_difference'], bins=20,
                 color=colors['correct'], edgecolor='white', linewidth=1.2, alpha=0.8)
    axes[1].axvline(true_positives['time_difference'].mean(), color=colors['dark'],
                    linestyle='--', linewidth=2, label=f'Mean: {true_positives["time_difference"].mean():.2f}s')
    axes[1].set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[1].set_title(f'CORRECT Predictions (n={len(true_positives)})\n(True Positives)',
                      fontsize=12, fontweight='bold', color=colors['correct'])
    axes[1].legend(fontsize=9)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_histograms.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_histograms.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_histograms.png/pdf")


def plot_combined_boxplot(false_negatives, true_positives, output_dir):
    """Create combined box plot with transparent data points."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = [false_negatives['time_difference'].values, true_positives['time_difference'].values]
    bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color=colors['dark'], linewidth=2),
                    whiskerprops=dict(color=colors['dark'], linewidth=1.5),
                    capprops=dict(color=colors['dark'], linewidth=1.5),
                    flierprops=dict(marker='', markersize=0))  # Hide outliers, we'll show all points
    
    # Color boxes
    bp['boxes'][0].set_facecolor(colors['wrong'])
    bp['boxes'][1].set_facecolor(colors['correct'])
    
    # Add transparent data points with jitter
    np.random.seed(42)
    x1_jitter = np.random.normal(1, 0.06, size=len(false_negatives))
    ax.scatter(x1_jitter, false_negatives['time_difference'], 
               c=colors['wrong'], s=40, alpha=0.4, edgecolors='black', linewidth=0.3, zorder=5)
    
    # Sample TP if too many points
    tp_sample = true_positives if len(true_positives) <= 100 else true_positives.sample(100, random_state=42)
    x2_jitter = np.random.normal(2, 0.06, size=len(tp_sample))
    ax.scatter(x2_jitter, tp_sample['time_difference'], 
               c=colors['correct'], s=40, alpha=0.4, edgecolors='black', linewidth=0.3, zorder=5)
    
    ax.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Difference Distribution: Wrong vs Correct Predictions',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Wrong\n(n={len(false_negatives)})', f'Correct\n(n={len(true_positives)})'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add mean markers
    ax.scatter([1], [false_negatives['time_difference'].mean()],
               marker='D', s=100, color=colors['dark'], zorder=10, label='Mean')
    ax.scatter([2], [true_positives['time_difference'].mean()],
               marker='D', s=100, color=colors['dark'], zorder=10)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_combined_boxplot.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_combined_boxplot.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_combined_boxplot.png/pdf")


def plot_scatter_comparison(false_negatives, true_positives, output_dir):
    """Create side-by-side scatter plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Time Difference vs Model Confidence: Wrong vs Correct Predictions',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Wrong predictions scatter
    scatter1 = axes[0].scatter(false_negatives['time_difference'], false_negatives['prob'],
                               c=false_negatives['prob'], cmap='RdYlGn',
                               s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[0].axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
    axes[0].set_title(f'WRONG Predictions (n={len(false_negatives)})\n(Lower prob = more confident wrong)',
                      fontsize=11, fontweight='bold', color=colors['wrong'])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    plt.colorbar(scatter1, ax=axes[0], label='Probability')
    
    # Correct predictions scatter (all points)
    scatter2 = axes[1].scatter(true_positives['time_difference'], true_positives['prob'],
                               c=true_positives['prob'], cmap='YlGn',
                               s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    axes[1].axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
    axes[1].set_title(f'CORRECT Predictions (n={len(true_positives)})\n(Higher prob = more confident)',
                      fontsize=11, fontweight='bold', color=colors['correct'])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.colorbar(scatter2, ax=axes[1], label='Probability')
    
    # Make y-axis limits the same
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_scatter_plots.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_scatter_plots.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_scatter_plots.png/pdf")


def plot_violin_comparison(false_negatives, true_positives, output_dir):
    """Create violin plot comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = ax.violinplot([false_negatives['time_difference'].values,
                           true_positives['time_difference'].values],
                          positions=[1, 2], showmeans=True, showmedians=True)
    
    # Color the violins
    parts['bodies'][0].set_facecolor(colors['wrong'])
    parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor(colors['correct'])
    parts['bodies'][1].set_alpha(0.6)
    
    ax.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Difference Distribution: Wrong vs Correct Predictions\n(Violin Plot)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Wrong\n(n={len(false_negatives)})', f'Correct\n(n={len(true_positives)})'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_violin_plot.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'comparison_violin_plot.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: comparison_violin_plot.png/pdf")


def plot_correct_predictions_analysis(true_positives, output_dir):
    """Create individual analysis plots for correct predictions."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Time Difference Analysis - CORRECT Predictions (True Positives)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(true_positives['time_difference'], bins=20,
             color=colors['correct'], edgecolor='white', linewidth=1.2, alpha=0.8)
    ax1.axvline(true_positives['time_difference'].mean(), color=colors['dark'],
                linestyle='--', linewidth=2, label=f'Mean: {true_positives["time_difference"].mean():.2f}s')
    ax1.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Time Differences\n(Event Time - Alert Time)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Box Plot
    ax2 = fig.add_subplot(2, 2, 2)
    bp = ax2.boxplot(true_positives['time_difference'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor=colors['correct'], alpha=0.6),
                     medianprops=dict(color=colors['dark'], linewidth=2),
                     whiskerprops=dict(color=colors['dark'], linewidth=1.5),
                     capprops=dict(color=colors['dark'], linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor=colors['accent'], markersize=6))
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Difference Distribution\n(Box Plot)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['True Positives'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    stats_text = (f"n = {len(true_positives)}\n"
                  f"Mean = {true_positives['time_difference'].mean():.2f}s\n"
                  f"Median = {true_positives['time_difference'].median():.2f}s\n"
                  f"Std = {true_positives['time_difference'].std():.2f}s")
    ax2.annotate(stats_text, xy=(1.3, true_positives['time_difference'].max()),
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Histogram of Event Times
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(true_positives['time_of_event'], bins=20, color=colors['correct'],
             edgecolor='white', linewidth=1.2, alpha=0.8)
    ax3.axvline(true_positives['time_of_event'].mean(), color=colors['dark'],
                linestyle='--', linewidth=2, label=f'Mean: {true_positives["time_of_event"].mean():.2f}s')
    ax3.set_xlabel('Time of Event (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Event Times', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Scatter Plot
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(true_positives['time_difference'], true_positives['prob'],
                          c=true_positives['prob'], cmap='YlGn',
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Time Difference vs Model Confidence\n(Correct Predictions)',
                  fontsize=14, fontweight='bold', pad=10)
    ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Decision boundary (0.5)')
    ax4.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Prediction Probability', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correct_predictions_analysis.png'), 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'correct_predictions_analysis.pdf'), 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: correct_predictions_analysis.png/pdf")


def main():
    print("=" * 80)
    print("TIME DIFFERENCE COMPARISON WITH RATIO LINE")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    fn_df, fp_df, tp_df, tn_df = load_data()
    
    # Filter for timing data
    false_negatives = fn_df.dropna(subset=['time_difference']).copy()
    true_positives = tp_df.dropna(subset=['time_difference']).copy()
    
    print(f"  False negatives with timing: {len(false_negatives)}")
    print(f"  True positives with timing: {len(true_positives)}")
    
    if len(false_negatives) == 0 or len(true_positives) == 0:
        print("\n⚠️  Not enough data for comparison")
        return
    
    # Print statistics
    print_statistics(false_negatives, true_positives)
    
    # Generate all plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    # 1. Overlapping histograms with ratio line
    bin_edges, bin_centers, n_wrong, n_correct, ratio, valid_mask = \
        plot_overlapping_histograms_with_ratio(false_negatives, true_positives, OUTPUT_DIR)
    
    # 2. Normalized overlapping histograms
    plot_normalized_overlapping_histograms(false_negatives, true_positives, bin_edges, bin_centers,
                                           ratio, valid_mask, OUTPUT_DIR)
    
    # 3. Side-by-side histograms
    plot_side_by_side_histograms(false_negatives, true_positives, OUTPUT_DIR)
    
    # 4. Combined boxplot
    plot_combined_boxplot(false_negatives, true_positives, OUTPUT_DIR)
    
    # 5. Scatter comparison
    plot_scatter_comparison(false_negatives, true_positives, OUTPUT_DIR)
    
    # 6. Violin plot
    plot_violin_comparison(false_negatives, true_positives, OUTPUT_DIR)
    
    # 7. Correct predictions individual analysis
    plot_correct_predictions_analysis(true_positives, OUTPUT_DIR)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nWrong Predictions (False Negatives):  {len(false_negatives)} samples")
    print(f"  Mean time diff: {false_negatives['time_difference'].mean():.3f}s")
    print(f"Correct Predictions (True Positives): {len(true_positives)} samples")
    print(f"  Mean time diff: {true_positives['time_difference'].mean():.3f}s")
    
    diff = false_negatives['time_difference'].mean() - true_positives['time_difference'].mean()
    print(f"\nDifference (Wrong - Correct): {diff:.3f}s")
    print(f"Wrong predictions have {'longer' if diff > 0 else 'shorter'} time differences on average")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
