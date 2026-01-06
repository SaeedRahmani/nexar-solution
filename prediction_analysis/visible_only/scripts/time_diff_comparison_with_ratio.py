"""
Time Difference Comparison with Ratio Line - VISIBLE ONLY SUBSET

This script creates overlapping histograms comparing time differences between
wrong and correct predictions, with a ratio line showing the wrong/correct ratio.

SUBSET: Excludes videos where "Target not visible" is True.

Usage:
    python prediction_analysis/visible_only/scripts/time_diff_comparison_with_ratio.py

Outputs to prediction_analysis/visible_only/comparative_analysis/:
    - comparison_overlapping_histograms.png/pdf (with ratio line)
    - comparison_overlapping_histograms_normalized.png/pdf
    - comparison_histograms.png/pdf
    - comparison_combined_boxplot.png/pdf
    - comparison_scatter_plots.png/pdf
    - comparison_violin_plot.png/pdf
    - correct_predictions_analysis.png/pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
BASE_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/visible_only")
WRONG_CSV = os.path.join(BASE_DIR, "wrong_predictions/all_wrong_predictions.csv")
CORRECT_CSV = os.path.join(BASE_DIR, "correct_predictions/all_correct_predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparative_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'wrong': '#E63946',
    'correct': '#4CAF50',
    'ratio': '#2E86AB',
}


def main():
    print("=" * 70)
    print("TIME DIFF COMPARISON WITH RATIO - VISIBLE ONLY SUBSET")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(WRONG_CSV) or not os.path.exists(CORRECT_CSV):
        print("⚠️  Required files not found. Run extract_predictions.py first.")
        return
    
    wrong_df = pd.read_csv(WRONG_CSV)
    correct_df = pd.read_csv(CORRECT_CSV)
    
    # Filter to only positive labels (accidents) for time difference comparison
    wrong_positive = wrong_df[wrong_df['true_label'] == 1]  # FN
    correct_positive = correct_df[correct_df['true_label'] == 1]  # TP
    
    print(f"\nData for time difference analysis:")
    print(f"  Wrong (FN - missed accidents): {len(wrong_positive)}")
    print(f"  Correct (TP - detected accidents): {len(correct_positive)}")
    
    if 'time_diff' not in wrong_df.columns:
        print("⚠️  No time_diff column found.")
        return
    
    # Get time differences
    wrong_time = wrong_positive[wrong_positive['time_diff'].notna() & (wrong_positive['time_diff'] > 0)]['time_diff']
    correct_time = correct_positive[correct_positive['time_diff'].notna() & (correct_positive['time_diff'] > 0)]['time_diff']
    
    print(f"  Wrong with valid time_diff: {len(wrong_time)}")
    print(f"  Correct with valid time_diff: {len(correct_time)}")
    
    if len(wrong_time) == 0 or len(correct_time) == 0:
        print("⚠️  Insufficient data for comparison.")
        return
    
    # ========================================
    # 1. Overlapping Histograms WITH RATIO LINE
    # ========================================
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Calculate bins
    all_times = pd.concat([wrong_time, correct_time])
    bins = np.linspace(0, min(all_times.max(), 5), 26)  # Cap at 5s for clarity
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot histograms
    wrong_counts, _ = np.histogram(wrong_time, bins=bins)
    correct_counts, _ = np.histogram(correct_time, bins=bins)
    
    ax1.bar(bin_centers, wrong_counts, width=bins[1]-bins[0], alpha=0.6, 
            label=f'Wrong (FN) n={len(wrong_time)}, mean={wrong_time.mean():.2f}s', 
            color=COLORS['wrong'], edgecolor='white')
    ax1.bar(bin_centers, correct_counts, width=bins[1]-bins[0], alpha=0.6, 
            label=f'Correct (TP) n={len(correct_time)}, mean={correct_time.mean():.2f}s', 
            color=COLORS['correct'], edgecolor='white')
    
    ax1.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add ratio line on secondary axis
    ax2 = ax1.twinx()
    
    # Calculate ratio (wrong / correct), avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(correct_counts > 0, wrong_counts / correct_counts, np.nan)
    
    # Only plot where we have valid ratios
    valid_mask = ~np.isnan(ratio) & (correct_counts >= 2)  # Require at least 2 correct for stability
    if valid_mask.any():
        ax2.plot(bin_centers[valid_mask], ratio[valid_mask], 
                 color=COLORS['ratio'], linewidth=3, marker='o', markersize=6,
                 label='Wrong/Correct Ratio')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1')
        ax2.set_ylabel('Wrong/Correct Ratio', fontsize=12, fontweight='bold', color=COLORS['ratio'])
        ax2.tick_params(axis='y', labelcolor=COLORS['ratio'])
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_ylim(0, max(ratio[valid_mask]) * 1.2 if valid_mask.any() else 2)
    
    plt.title('Time Difference: Wrong vs Correct Predictions (with Ratio Line)\n[VISIBLE ONLY SUBSET]', 
              fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_overlapping_histograms.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_overlapping_histograms.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n✅ Saved: comparison_overlapping_histograms.png/pdf (with ratio line)")
    
    # ========================================
    # 2. Normalized Overlapping Histograms
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(wrong_time, bins=bins, alpha=0.6, density=True,
            label=f'Wrong (FN) n={len(wrong_time)}', color=COLORS['wrong'], edgecolor='white')
    ax.hist(correct_time, bins=bins, alpha=0.6, density=True,
            label=f'Correct (TP) n={len(correct_time)}', color=COLORS['correct'], edgecolor='white')
    
    ax.axvline(wrong_time.mean(), color=COLORS['wrong'], linestyle='--', linewidth=2,
               label=f'Wrong mean: {wrong_time.mean():.2f}s')
    ax.axvline(correct_time.mean(), color=COLORS['correct'], linestyle='--', linewidth=2,
               label=f'Correct mean: {correct_time.mean():.2f}s')
    
    ax.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Time Difference Distribution (Normalized)\n[VISIBLE ONLY SUBSET]', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_overlapping_histograms_normalized.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_overlapping_histograms_normalized.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: comparison_overlapping_histograms_normalized.png/pdf")
    
    # ========================================
    # 3. Side-by-side Histograms
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Time Difference Analysis\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
    
    axes[0].hist(wrong_time, bins=20, color=COLORS['wrong'], edgecolor='white', alpha=0.8)
    axes[0].axvline(wrong_time.mean(), color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time Difference (s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Wrong (FN): mean={wrong_time.mean():.2f}s, n={len(wrong_time)}')
    
    axes[1].hist(correct_time, bins=20, color=COLORS['correct'], edgecolor='white', alpha=0.8)
    axes[1].axvline(correct_time.mean(), color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Time Difference (s)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Correct (TP): mean={correct_time.mean():.2f}s, n={len(correct_time)}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_histograms.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_histograms.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: comparison_histograms.png/pdf")
    
    # ========================================
    # 4. Combined Box Plot
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot([wrong_time, correct_time], patch_artist=True, 
                    labels=['Wrong (FN)', 'Correct (TP)'],
                    flierprops=dict(marker='', markersize=0))  # Hide outliers, we'll show all points
    bp['boxes'][0].set_facecolor(COLORS['wrong'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['correct'])
    bp['boxes'][1].set_alpha(0.7)
    
    # Add transparent data points with jitter
    np.random.seed(42)
    x1_jitter = np.random.normal(1, 0.06, size=len(wrong_time))
    ax.scatter(x1_jitter, wrong_time, 
               c=COLORS['wrong'], s=40, alpha=0.4, edgecolors='black', linewidth=0.3, zorder=5)
    
    # Sample TP if too many points
    if len(correct_time) > 100:
        sample_idx = np.random.choice(len(correct_time), 100, replace=False)
        correct_sample = correct_time.iloc[sample_idx] if hasattr(correct_time, 'iloc') else correct_time[sample_idx]
    else:
        correct_sample = correct_time
    x2_jitter = np.random.normal(2, 0.06, size=len(correct_sample))
    ax.scatter(x2_jitter, correct_sample, 
               c=COLORS['correct'], s=40, alpha=0.4, edgecolors='black', linewidth=0.3, zorder=5)
    
    ax.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Difference Comparison\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
    
    # Add mean markers
    means = [wrong_time.mean(), correct_time.mean()]
    ax.scatter([1, 2], means, color='black', marker='D', s=50, zorder=10, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_combined_boxplot.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_combined_boxplot.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: comparison_combined_boxplot.png/pdf")
    
    # ========================================
    # 5. Violin Plot
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = ax.violinplot([wrong_time, correct_time], positions=[1, 2], 
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(COLORS['wrong'])
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor(COLORS['correct'])
    parts['bodies'][1].set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Wrong (FN)', 'Correct (TP)'])
    ax.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Difference Distribution (Violin Plot)\n[VISIBLE ONLY SUBSET]', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_violin_plot.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_violin_plot.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: comparison_violin_plot.png/pdf")
    
    # ========================================
    # 6. Scatter Plot
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Time Difference Scatter Plots\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
    
    # Wrong predictions
    axes[0].scatter(range(len(wrong_time)), wrong_time.values, 
                    c=COLORS['wrong'], alpha=0.6, s=30)
    axes[0].axhline(wrong_time.mean(), color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Time Difference (s)')
    axes[0].set_title(f'Wrong (FN): n={len(wrong_time)}')
    
    # Correct predictions
    axes[1].scatter(range(len(correct_time)), correct_time.values, 
                    c=COLORS['correct'], alpha=0.6, s=30)
    axes[1].axhline(correct_time.mean(), color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Time Difference (s)')
    axes[1].set_title(f'Correct (TP): n={len(correct_time)}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_scatter_plots.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_scatter_plots.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: comparison_scatter_plots.png/pdf")
    
    # ========================================
    # 7. Correct Predictions Only Analysis
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Correct Predictions (TP) Analysis\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
    
    # Histogram
    axes[0].hist(correct_time, bins=25, color=COLORS['correct'], edgecolor='white', alpha=0.8)
    axes[0].axvline(correct_time.mean(), color='black', linestyle='--', linewidth=2,
                    label=f'Mean: {correct_time.mean():.2f}s')
    axes[0].axvline(correct_time.median(), color='gray', linestyle=':', linewidth=2,
                    label=f'Median: {correct_time.median():.2f}s')
    axes[0].set_xlabel('Time Difference (s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Time Difference Distribution')
    axes[0].legend()
    
    # CDF
    sorted_time = np.sort(correct_time)
    cdf = np.arange(1, len(sorted_time) + 1) / len(sorted_time)
    axes[1].plot(sorted_time, cdf, color=COLORS['correct'], linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(correct_time.median(), color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Difference (s)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correct_predictions_analysis.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'correct_predictions_analysis.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: correct_predictions_analysis.png/pdf")
    
    # Statistical summary
    u_stat, p_value = stats.mannwhitneyu(wrong_time, correct_time, alternative='two-sided')
    
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY - VISIBLE ONLY SUBSET")
    print("=" * 70)
    print(f"\nWrong (FN): mean={wrong_time.mean():.3f}s, median={wrong_time.median():.3f}s, n={len(wrong_time)}")
    print(f"Correct (TP): mean={correct_time.mean():.3f}s, median={correct_time.median():.3f}s, n={len(correct_time)}")
    print(f"\nMann-Whitney U test: p={p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
