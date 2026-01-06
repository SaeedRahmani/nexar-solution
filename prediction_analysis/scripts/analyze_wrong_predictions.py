"""
Analyze wrong predictions - focusing on false negatives (missed accidents).

This script:
1. Analyzes time differences for false negatives
2. Analyzes prediction confidence for wrong predictions
3. Creates visualizations

Usage:
    python prediction_analysis/scripts/analyze_wrong_predictions.py

Outputs to prediction_analysis/wrong_predictions/:
    - time_difference_analysis.png/pdf
    - prediction_confidence_analysis.png/pdf
    - summary_statistics.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = "/home/sra2157/git/nexar-solution"
sys.path.insert(0, PROJECT_ROOT)

# Paths
INPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/wrong_predictions")
OUTPUT_DIR = INPUT_DIR  # Output to same folder

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'fn': '#C73E1D',      # Red for false negatives (critical)
    'fp': '#F18F01',      # Orange for false positives
    'primary': '#2E86AB', # Blue
    'secondary': '#A23B72',
    'accent': '#4CAF50',  # Green
    'dark': '#3B1F2B'
}


def load_data():
    """Load wrong predictions data."""
    fn_path = os.path.join(INPUT_DIR, 'false_negatives.csv')
    fp_path = os.path.join(INPUT_DIR, 'false_positives.csv')
    
    if not os.path.exists(fn_path) or not os.path.exists(fp_path):
        print("Error: Wrong predictions files not found!")
        print("Please run extract_predictions.py first.")
        sys.exit(1)
    
    fn_df = pd.read_csv(fn_path, dtype={'video_id': str})
    fp_df = pd.read_csv(fp_path, dtype={'video_id': str})
    
    return fn_df, fp_df


def analyze_false_negatives(fn_df):
    """Analyze false negatives with timing data."""
    
    # Filter to those with timing data
    fn_with_timing = fn_df.dropna(subset=['time_of_event', 'time_of_alert'])
    
    if len(fn_with_timing) == 0:
        print("No false negatives with timing data found.")
        return None
    
    stats = {
        'count': len(fn_with_timing),
        'mean_time_diff': fn_with_timing['time_difference'].mean(),
        'std_time_diff': fn_with_timing['time_difference'].std(),
        'min_time_diff': fn_with_timing['time_difference'].min(),
        'max_time_diff': fn_with_timing['time_difference'].max(),
        'median_time_diff': fn_with_timing['time_difference'].median(),
        'mean_prob': fn_with_timing['prob'].mean(),
        'train_count': len(fn_with_timing[fn_with_timing['split'] == 'train']),
        'val_count': len(fn_with_timing[fn_with_timing['split'] == 'val'])
    }
    
    return stats, fn_with_timing


def plot_false_negatives_analysis(fn_df, output_dir):
    """Create visualization for false negatives."""
    
    fn_with_timing = fn_df.dropna(subset=['time_of_event', 'time_of_alert'])
    
    if len(fn_with_timing) < 2:
        print("Not enough false negatives for detailed plotting.")
        return
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('FALSE NEGATIVES ANALYSIS\n(Missed Accidents - CRITICAL)', 
                 fontsize=16, fontweight='bold', color=COLORS['fn'])
    
    # Plot 1: Histogram of Time Differences
    ax1 = fig.add_subplot(2, 2, 1)
    n, bins, patches = ax1.hist(fn_with_timing['time_difference'], bins=10, 
                                 color=COLORS['fn'], edgecolor='white', 
                                 linewidth=1.2, alpha=0.8)
    ax1.axvline(fn_with_timing['time_difference'].mean(), color='black', 
                linestyle='--', linewidth=2, 
                label=f'Mean: {fn_with_timing["time_difference"].mean():.2f}s')
    ax1.set_xlabel('Time Difference (Event - Alert) in seconds', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Time Differences', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Box Plot with individual points
    ax2 = fig.add_subplot(2, 2, 2)
    bp = ax2.boxplot(fn_with_timing['time_difference'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['fn'], alpha=0.6),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color=COLORS['dark'], linewidth=1.5),
                     capprops=dict(color=COLORS['dark'], linewidth=1.5))
    
    x_jitter = np.random.normal(1, 0.04, size=len(fn_with_timing))
    ax2.scatter(x_jitter, fn_with_timing['time_difference'], 
                c=COLORS['fp'], s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Difference Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['False Negatives'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Stats annotation
    stats_text = (f"n = {len(fn_with_timing)}\n"
                  f"Mean = {fn_with_timing['time_difference'].mean():.2f}s\n"
                  f"Median = {fn_with_timing['time_difference'].median():.2f}s\n"
                  f"Std = {fn_with_timing['time_difference'].std():.2f}s")
    ax2.annotate(stats_text, xy=(1.3, fn_with_timing['time_difference'].max()), 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Bar chart by video
    ax3 = fig.add_subplot(2, 2, 3)
    sorted_fn = fn_with_timing.sort_values('time_difference', ascending=True)
    colors_by_split = [COLORS['fn'] if s == 'val' else COLORS['secondary'] for s in sorted_fn['split']]
    bars = ax3.barh(sorted_fn['video_id'], sorted_fn['time_difference'], 
                    color=colors_by_split, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax3.axvline(fn_with_timing['time_difference'].mean(), color='black', 
                linestyle='--', linewidth=2, label=f'Mean: {fn_with_timing["time_difference"].mean():.2f}s')
    ax3.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Video ID', fontsize=12, fontweight='bold')
    ax3.set_title('Time Difference by Video\n(Red=Val, Purple=Train)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Scatter - Time Diff vs Probability
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(fn_with_timing['time_difference'], fn_with_timing['prob'], 
                          c=fn_with_timing['prob'], cmap='RdYlGn', 
                          s=150, alpha=0.8, edgecolors='black', linewidth=1)
    
    for _, row in fn_with_timing.iterrows():
        ax4.annotate(row['video_id'], (row['time_difference'], row['prob']),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)
    
    ax4.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Time Difference vs Model Confidence\n(Red = confident wrong = BAD)', 
                  fontsize=14, fontweight='bold')
    ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Decision boundary (0.5)')
    ax4.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Prediction Probability', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(os.path.join(output_dir, 'time_difference_analysis.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'time_difference_analysis.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: time_difference_analysis.png/pdf")


def plot_confidence_analysis(fn_df, fp_df, output_dir):
    """Analyze prediction confidence for wrong predictions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('PREDICTION CONFIDENCE ANALYSIS\n(Wrong Predictions)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram of FN probabilities
    ax1 = axes[0]
    ax1.hist(fn_df['prob'], bins=15, color=COLORS['fn'], edgecolor='white', alpha=0.8)
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax1.axvline(fn_df['prob'].mean(), color='white', linestyle='-', linewidth=2, 
                label=f'Mean: {fn_df["prob"].mean():.3f}')
    ax1.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title(f'False Negatives (n={len(fn_df)})\n(Lower prob = more confident mistake)', 
                  fontsize=12, fontweight='bold', color=COLORS['fn'])
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    
    # Plot 2: Histogram of FP probabilities
    ax2 = axes[1]
    ax2.hist(fp_df['prob'], bins=15, color=COLORS['fp'], edgecolor='white', alpha=0.8)
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax2.axvline(fp_df['prob'].mean(), color='white', linestyle='-', linewidth=2, 
                label=f'Mean: {fp_df["prob"].mean():.3f}')
    ax2.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title(f'False Positives (n={len(fp_df)})\n(Higher prob = more confident mistake)', 
                  fontsize=12, fontweight='bold', color=COLORS['fp'])
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    
    # Plot 3: Combined box plot
    ax3 = axes[2]
    data = [fn_df['prob'].values, fp_df['prob'].values]
    bp = ax3.boxplot(data, patch_artist=True, labels=['False Neg\n(Missed)', 'False Pos\n(False Alarm)'])
    bp['boxes'][0].set_facecolor(COLORS['fn'])
    bp['boxes'][1].set_facecolor(COLORS['fp'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax3.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision boundary')
    ax3.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Confidence Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save
    plt.savefig(os.path.join(output_dir, 'prediction_confidence_analysis.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'prediction_confidence_analysis.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: prediction_confidence_analysis.png/pdf")


def save_summary(fn_df, fp_df, output_dir):
    """Save summary statistics to text file."""
    
    fn_with_timing = fn_df.dropna(subset=['time_of_event', 'time_of_alert'])
    
    summary = []
    summary.append("=" * 70)
    summary.append("WRONG PREDICTIONS ANALYSIS SUMMARY")
    summary.append("=" * 70)
    
    summary.append(f"\nTotal wrong predictions: {len(fn_df) + len(fp_df)}")
    summary.append(f"  - False Negatives: {len(fn_df)} (missed accidents - CRITICAL)")
    summary.append(f"  - False Positives: {len(fp_df)} (false alarms)")
    
    summary.append("\n" + "-" * 70)
    summary.append("FALSE NEGATIVES (Missed Accidents)")
    summary.append("-" * 70)
    
    summary.append(f"\nBy split:")
    summary.append(f"  - Training: {len(fn_df[fn_df['split'] == 'train'])}")
    summary.append(f"  - Validation: {len(fn_df[fn_df['split'] == 'val'])}")
    
    summary.append(f"\nPrediction confidence:")
    summary.append(f"  - Mean probability: {fn_df['prob'].mean():.4f}")
    summary.append(f"  - Std: {fn_df['prob'].std():.4f}")
    summary.append(f"  - Min: {fn_df['prob'].min():.4f}")
    summary.append(f"  - Max: {fn_df['prob'].max():.4f}")
    
    if len(fn_with_timing) > 0:
        summary.append(f"\nTime difference (event - alert) for {len(fn_with_timing)} videos:")
        summary.append(f"  - Mean: {fn_with_timing['time_difference'].mean():.3f}s")
        summary.append(f"  - Std: {fn_with_timing['time_difference'].std():.3f}s")
        summary.append(f"  - Min: {fn_with_timing['time_difference'].min():.3f}s")
        summary.append(f"  - Max: {fn_with_timing['time_difference'].max():.3f}s")
        summary.append(f"  - Median: {fn_with_timing['time_difference'].median():.3f}s")
    
    summary.append("\n" + "-" * 70)
    summary.append("FALSE POSITIVES (False Alarms)")
    summary.append("-" * 70)
    
    summary.append(f"\nBy split:")
    summary.append(f"  - Training: {len(fp_df[fp_df['split'] == 'train'])}")
    summary.append(f"  - Validation: {len(fp_df[fp_df['split'] == 'val'])}")
    
    summary.append(f"\nPrediction confidence:")
    summary.append(f"  - Mean probability: {fp_df['prob'].mean():.4f}")
    summary.append(f"  - Std: {fp_df['prob'].std():.4f}")
    summary.append(f"  - Min: {fp_df['prob'].min():.4f}")
    summary.append(f"  - Max: {fp_df['prob'].max():.4f}")
    
    summary.append("\n" + "=" * 70)
    
    # Save
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"  ✅ Saved: summary_statistics.txt")
    
    # Also print
    print('\n'.join(summary))


def main():
    print("=" * 70)
    print("ANALYZING WRONG PREDICTIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    fn_df, fp_df = load_data()
    print(f"  - False negatives: {len(fn_df)}")
    print(f"  - False positives: {len(fp_df)}")
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_false_negatives_analysis(fn_df, OUTPUT_DIR)
    plot_confidence_analysis(fn_df, fp_df, OUTPUT_DIR)
    
    # Save summary
    print("\nGenerating summary...")
    save_summary(fn_df, fp_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
