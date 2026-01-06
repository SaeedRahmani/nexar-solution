"""
Comparative Analysis: Wrong vs Correct Predictions

This script:
1. Compares time differences between false negatives (wrong) and true positives (correct)
2. Tests statistical significance of differences
3. Creates side-by-side visualizations

Key question: Do missed accidents have different timing characteristics than detected accidents?

Usage:
    python prediction_analysis/scripts/comparative_analysis.py

Outputs to prediction_analysis/comparative_analysis/:
    - time_diff_comparison_histograms.png/pdf
    - time_diff_comparison_boxplots.png/pdf
    - time_diff_comparison_violin.png/pdf
    - confidence_comparison.png/pdf
    - statistical_comparison.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats

# Add project root to path
PROJECT_ROOT = "/home/sra2157/git/nexar-solution"
sys.path.insert(0, PROJECT_ROOT)

# Paths
WRONG_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/wrong_predictions")
CORRECT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/correct_predictions")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/comparative_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'wrong': '#C73E1D',    # Red for wrong
    'correct': '#4CAF50',  # Green for correct
    'fn': '#C73E1D',
    'fp': '#F18F01',
    'tp': '#4CAF50',
    'tn': '#2E86AB',
}


def load_data():
    """Load all prediction data."""
    fn_df = pd.read_csv(os.path.join(WRONG_DIR, 'false_negatives.csv'), dtype={'video_id': str})
    fp_df = pd.read_csv(os.path.join(WRONG_DIR, 'false_positives.csv'), dtype={'video_id': str})
    tp_df = pd.read_csv(os.path.join(CORRECT_DIR, 'true_positives.csv'), dtype={'video_id': str})
    tn_df = pd.read_csv(os.path.join(CORRECT_DIR, 'true_negatives.csv'), dtype={'video_id': str})
    
    return fn_df, fp_df, tp_df, tn_df


def compare_time_differences(fn_df, tp_df):
    """Compare time differences between FN and TP."""
    
    fn_timing = fn_df.dropna(subset=['time_difference'])
    tp_timing = tp_df.dropna(subset=['time_difference'])
    
    return fn_timing, tp_timing


def plot_histogram_comparison(fn_timing, tp_timing, output_dir):
    """Create side-by-side and overlapping histograms."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('TIME DIFFERENCE COMPARISON\nFalse Negatives (Missed) vs True Positives (Detected)', 
                 fontsize=14, fontweight='bold')
    
    # Common bins
    all_data = np.concatenate([fn_timing['time_difference'], tp_timing['time_difference']])
    bins = np.linspace(all_data.min(), all_data.max(), 20)
    
    # Plot 1: Side-by-side
    ax1 = axes[0]
    ax1.hist(fn_timing['time_difference'], bins=bins, color=COLORS['wrong'], 
             edgecolor='white', alpha=0.8, label=f'False Neg (n={len(fn_timing)})')
    ax1.axvline(fn_timing['time_difference'].mean(), color=COLORS['wrong'], 
                linestyle='--', linewidth=2)
    ax1.set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('False Negatives (Missed Accidents)', fontsize=12, fontweight='bold', color=COLORS['wrong'])
    ax1.legend()
    
    ax2 = axes[1]
    ax2.hist(tp_timing['time_difference'], bins=bins, color=COLORS['correct'], 
             edgecolor='white', alpha=0.8, label=f'True Pos (n={len(tp_timing)})')
    ax2.axvline(tp_timing['time_difference'].mean(), color=COLORS['correct'], 
                linestyle='--', linewidth=2)
    ax2.set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('True Positives (Detected Accidents)', fontsize=12, fontweight='bold', color=COLORS['correct'])
    ax2.legend()
    
    # Plot 3: Overlapping
    ax3 = axes[2]
    ax3.hist(fn_timing['time_difference'], bins=bins, color=COLORS['wrong'], 
             edgecolor='white', alpha=0.6, label=f'Missed (n={len(fn_timing)}, μ={fn_timing["time_difference"].mean():.2f}s)')
    ax3.hist(tp_timing['time_difference'], bins=bins, color=COLORS['correct'], 
             edgecolor='white', alpha=0.6, label=f'Detected (n={len(tp_timing)}, μ={tp_timing["time_difference"].mean():.2f}s)')
    ax3.axvline(fn_timing['time_difference'].mean(), color=COLORS['wrong'], linestyle='--', linewidth=2)
    ax3.axvline(tp_timing['time_difference'].mean(), color=COLORS['correct'], linestyle='--', linewidth=2)
    ax3.set_xlabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Overlapping Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_histograms.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_histograms.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: time_diff_comparison_histograms.png/pdf")


def plot_boxplot_comparison(fn_timing, tp_timing, output_dir):
    """Create box plot comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('TIME DIFFERENCE DISTRIBUTION COMPARISON', fontsize=14, fontweight='bold')
    
    # Plot 1: Side-by-side box plots
    ax1 = axes[0]
    data = [fn_timing['time_difference'].values, tp_timing['time_difference'].values]
    bp = ax1.boxplot(data, patch_artist=True, labels=['False Neg\n(Missed)', 'True Pos\n(Detected)'])
    bp['boxes'][0].set_facecolor(COLORS['wrong'])
    bp['boxes'][1].set_facecolor(COLORS['correct'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax1.set_ylabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    
    # Add mean markers
    means = [fn_timing['time_difference'].mean(), tp_timing['time_difference'].mean()]
    ax1.scatter([1, 2], means, marker='D', color='black', s=50, zorder=5, label='Mean')
    ax1.legend()
    
    # Plot 2: Strip plot overlay
    ax2 = axes[1]
    bp2 = ax2.boxplot(data, patch_artist=True, labels=['False Neg\n(Missed)', 'True Pos\n(Detected)'])
    bp2['boxes'][0].set_facecolor(COLORS['wrong'])
    bp2['boxes'][1].set_facecolor(COLORS['correct'])
    for box in bp2['boxes']:
        box.set_alpha(0.5)
    
    # Add individual points
    x1_jitter = np.random.normal(1, 0.08, size=len(fn_timing))
    ax2.scatter(x1_jitter, fn_timing['time_difference'], 
                c=COLORS['wrong'], s=50, alpha=0.6, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Sample TP for visibility
    tp_sample = tp_timing.sample(min(100, len(tp_timing)))
    x2_jitter = np.random.normal(2, 0.08, size=len(tp_sample))
    ax2.scatter(x2_jitter, tp_sample['time_difference'], 
                c=COLORS['correct'], s=30, alpha=0.4, edgecolors='black', linewidth=0.3, zorder=5)
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot with Individual Points\n(TP sampled for clarity)', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_boxplots.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_boxplots.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: time_diff_comparison_boxplots.png/pdf")


def plot_violin_comparison(fn_timing, tp_timing, output_dir):
    """Create violin plot comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('TIME DIFFERENCE DISTRIBUTION\n(Violin Plot Comparison)', fontsize=14, fontweight='bold')
    
    # Prepare data for violin plot
    data = [fn_timing['time_difference'].values, tp_timing['time_difference'].values]
    positions = [1, 2]
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    # Color the violins
    colors = [COLORS['wrong'], COLORS['correct']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Style the lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)
    
    # Add scatter points
    x1_jitter = np.random.normal(1, 0.05, size=len(fn_timing))
    ax.scatter(x1_jitter, fn_timing['time_difference'], 
               c=COLORS['wrong'], s=30, alpha=0.5, edgecolors='black', linewidth=0.3)
    
    tp_sample = tp_timing.sample(min(50, len(tp_timing)))
    x2_jitter = np.random.normal(2, 0.05, size=len(tp_sample))
    ax.scatter(x2_jitter, tp_sample['time_difference'], 
               c=COLORS['correct'], s=30, alpha=0.5, edgecolors='black', linewidth=0.3)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'False Negatives\n(Missed, n={len(fn_timing)})', 
                        f'True Positives\n(Detected, n={len(tp_timing)})'])
    ax.set_ylabel('Time Difference (seconds)', fontsize=11, fontweight='bold')
    
    # Add statistics annotation
    diff = tp_timing['time_difference'].mean() - fn_timing['time_difference'].mean()
    ax.text(1.5, ax.get_ylim()[1] * 0.95, 
            f'Mean difference: {diff:.3f}s\n(Detected accidents have\nlarger time gaps)',
            ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_violin.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'time_diff_comparison_violin.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: time_diff_comparison_violin.png/pdf")


def plot_confidence_comparison(fn_df, fp_df, tp_df, tn_df, output_dir):
    """Compare prediction confidence across all categories."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PREDICTION CONFIDENCE COMPARISON\n(All Prediction Categories)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Box plots
    ax1 = axes[0]
    data = [fn_df['prob'].values, fp_df['prob'].values, tp_df['prob'].values, tn_df['prob'].values]
    labels = ['False Neg\n(Missed)', 'False Pos\n(False Alarm)', 'True Pos\n(Detected)', 'True Neg\n(Safe)']
    colors_list = [COLORS['fn'], COLORS['fp'], COLORS['tp'], COLORS['tn']]
    
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    for i, (box, color) in enumerate(zip(bp['boxes'], colors_list)):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision boundary')
    ax1.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_title('Confidence by Category', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Plot 2: Histogram comparison - wrong vs correct
    ax2 = axes[1]
    wrong_probs = np.concatenate([fn_df['prob'].values, fp_df['prob'].values])
    correct_probs = np.concatenate([tp_df['prob'].values, tn_df['prob'].values])
    
    bins = np.linspace(0, 1, 25)
    ax2.hist(wrong_probs, bins=bins, color=COLORS['wrong'], edgecolor='white', 
             alpha=0.6, label=f'Wrong predictions (n={len(wrong_probs)})')
    ax2.hist(correct_probs, bins=bins, color=COLORS['correct'], edgecolor='white', 
             alpha=0.6, label=f'Correct predictions (n={len(correct_probs)})')
    ax2.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision boundary')
    ax2.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: confidence_comparison.png/pdf")


def statistical_analysis(fn_timing, tp_timing, fn_df, fp_df, tp_df, tn_df, output_dir):
    """Perform statistical tests and save results."""
    
    results = []
    results.append("=" * 70)
    results.append("STATISTICAL COMPARISON ANALYSIS")
    results.append("=" * 70)
    
    # Time difference comparison
    results.append("\n" + "-" * 70)
    results.append("TIME DIFFERENCE COMPARISON")
    results.append("(False Negatives vs True Positives)")
    results.append("-" * 70)
    
    results.append(f"\nFalse Negatives (Missed Accidents):")
    results.append(f"  n = {len(fn_timing)}")
    results.append(f"  Mean = {fn_timing['time_difference'].mean():.3f}s")
    results.append(f"  Std = {fn_timing['time_difference'].std():.3f}s")
    results.append(f"  Median = {fn_timing['time_difference'].median():.3f}s")
    
    results.append(f"\nTrue Positives (Detected Accidents):")
    results.append(f"  n = {len(tp_timing)}")
    results.append(f"  Mean = {tp_timing['time_difference'].mean():.3f}s")
    results.append(f"  Std = {tp_timing['time_difference'].std():.3f}s")
    results.append(f"  Median = {tp_timing['time_difference'].median():.3f}s")
    
    # Mann-Whitney U test (non-parametric)
    statistic, pvalue = stats.mannwhitneyu(
        fn_timing['time_difference'], 
        tp_timing['time_difference'],
        alternative='two-sided'
    )
    
    results.append(f"\nMann-Whitney U Test (non-parametric):")
    results.append(f"  U-statistic: {statistic:.4f}")
    results.append(f"  P-value: {pvalue:.6f}")
    results.append(f"  Significant (p<0.05): {'Yes' if pvalue < 0.05 else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((fn_timing['time_difference'].std()**2 + tp_timing['time_difference'].std()**2) / 2)
    cohens_d = (fn_timing['time_difference'].mean() - tp_timing['time_difference'].mean()) / pooled_std
    
    results.append(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_interp = "Small"
    elif abs(cohens_d) < 0.5:
        effect_interp = "Small-Medium"
    elif abs(cohens_d) < 0.8:
        effect_interp = "Medium"
    else:
        effect_interp = "Large"
    results.append(f"  Interpretation: {effect_interp}")
    
    # Interpretation
    results.append("\n" + "-" * 70)
    results.append("INTERPRETATION")
    results.append("-" * 70)
    
    diff = tp_timing['time_difference'].mean() - fn_timing['time_difference'].mean()
    results.append(f"\nMean time difference:")
    results.append(f"  Missed accidents: {fn_timing['time_difference'].mean():.3f}s")
    results.append(f"  Detected accidents: {tp_timing['time_difference'].mean():.3f}s")
    results.append(f"  Difference: {diff:.3f}s")
    
    if pvalue < 0.05:
        results.append(f"\n⚠️  SIGNIFICANT FINDING:")
        if diff > 0:
            results.append(f"    Missed accidents have SHORTER time gaps (event closer to alert).")
            results.append(f"    This suggests the model struggles with last-moment events.")
        else:
            results.append(f"    Missed accidents have LONGER time gaps.")
            results.append(f"    The model may be missing slow-developing incidents.")
    else:
        results.append(f"\n✅ No significant difference in time gaps between missed and detected accidents.")
    
    # Confidence comparison
    results.append("\n" + "-" * 70)
    results.append("CONFIDENCE COMPARISON")
    results.append("-" * 70)
    
    results.append(f"\nMean prediction probability by category:")
    results.append(f"  False Negatives: {fn_df['prob'].mean():.4f} (lower = more confident wrong)")
    results.append(f"  False Positives: {fp_df['prob'].mean():.4f} (higher = more confident wrong)")
    results.append(f"  True Positives:  {tp_df['prob'].mean():.4f}")
    results.append(f"  True Negatives:  {tn_df['prob'].mean():.4f}")
    
    results.append("\n" + "=" * 70)
    
    # Save
    with open(os.path.join(output_dir, 'statistical_comparison.txt'), 'w') as f:
        f.write('\n'.join(results))
    
    print(f"  ✅ Saved: statistical_comparison.txt")
    
    # Print
    print('\n'.join(results))


def main():
    print("=" * 70)
    print("COMPARATIVE ANALYSIS: WRONG vs CORRECT PREDICTIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    fn_df, fp_df, tp_df, tn_df = load_data()
    print(f"  - False Negatives: {len(fn_df)}")
    print(f"  - False Positives: {len(fp_df)}")
    print(f"  - True Positives: {len(tp_df)}")
    print(f"  - True Negatives: {len(tn_df)}")
    
    # Get timing data
    fn_timing, tp_timing = compare_time_differences(fn_df, tp_df)
    print(f"\nWith timing data:")
    print(f"  - False Negatives: {len(fn_timing)}")
    print(f"  - True Positives: {len(tp_timing)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_histogram_comparison(fn_timing, tp_timing, OUTPUT_DIR)
    plot_boxplot_comparison(fn_timing, tp_timing, OUTPUT_DIR)
    plot_violin_comparison(fn_timing, tp_timing, OUTPUT_DIR)
    plot_confidence_comparison(fn_df, fp_df, tp_df, tn_df, OUTPUT_DIR)
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    statistical_analysis(fn_timing, tp_timing, fn_df, fp_df, tp_df, tn_df, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
