"""
Analyze correct predictions - focusing on true positives.

This script:
1. Analyzes time differences for true positives
2. Analyzes prediction confidence for correct predictions
3. Creates visualizations

Usage:
    python prediction_analysis/scripts/analyze_correct_predictions.py

Outputs to prediction_analysis/correct_predictions/:
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
INPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/correct_predictions")
OUTPUT_DIR = INPUT_DIR

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'tp': '#4CAF50',      # Green for true positives
    'tn': '#2E86AB',      # Blue for true negatives
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'dark': '#3B1F2B'
}


def load_data():
    """Load correct predictions data."""
    tp_path = os.path.join(INPUT_DIR, 'true_positives.csv')
    tn_path = os.path.join(INPUT_DIR, 'true_negatives.csv')
    
    if not os.path.exists(tp_path) or not os.path.exists(tn_path):
        print("Error: Correct predictions files not found!")
        print("Please run extract_predictions.py first.")
        sys.exit(1)
    
    tp_df = pd.read_csv(tp_path, dtype={'video_id': str})
    tn_df = pd.read_csv(tn_path, dtype={'video_id': str})
    
    return tp_df, tn_df


def plot_true_positives_analysis(tp_df, output_dir):
    """Create visualization for true positives."""
    
    tp_with_timing = tp_df.dropna(subset=['time_of_event', 'time_of_alert'])
    
    if len(tp_with_timing) < 2:
        print("Not enough true positives with timing for detailed plotting.")
        return
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('TRUE POSITIVES ANALYSIS\n(Correctly Detected Accidents)', 
                 fontsize=16, fontweight='bold', color=COLORS['tp'])
    
    # Plot 1: Histogram of Time Differences
    ax1 = fig.add_subplot(2, 2, 1)
    n, bins, patches = ax1.hist(tp_with_timing['time_difference'], bins=20, 
                                 color=COLORS['tp'], edgecolor='white', 
                                 linewidth=1.2, alpha=0.8)
    ax1.axvline(tp_with_timing['time_difference'].mean(), color='black', 
                linestyle='--', linewidth=2, 
                label=f'Mean: {tp_with_timing["time_difference"].mean():.2f}s')
    ax1.set_xlabel('Time Difference (Event - Alert) in seconds', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Time Differences', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Box Plot with individual points (sample)
    ax2 = fig.add_subplot(2, 2, 2)
    bp = ax2.boxplot(tp_with_timing['time_difference'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['tp'], alpha=0.6),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color=COLORS['dark'], linewidth=1.5),
                     capprops=dict(color=COLORS['dark'], linewidth=1.5))
    
    # Sample points to avoid overcrowding
    sample_size = min(50, len(tp_with_timing))
    sample = tp_with_timing.sample(sample_size) if len(tp_with_timing) > sample_size else tp_with_timing
    x_jitter = np.random.normal(1, 0.04, size=len(sample))
    ax2.scatter(x_jitter, sample['time_difference'], 
                c=COLORS['accent'], s=40, alpha=0.5, edgecolors='black', linewidth=0.3, zorder=5)
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Time Difference Distribution\n(showing {sample_size} points)', fontsize=14, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['True Positives'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Stats annotation
    stats_text = (f"n = {len(tp_with_timing)}\n"
                  f"Mean = {tp_with_timing['time_difference'].mean():.2f}s\n"
                  f"Median = {tp_with_timing['time_difference'].median():.2f}s\n"
                  f"Std = {tp_with_timing['time_difference'].std():.2f}s")
    ax2.annotate(stats_text, xy=(1.3, tp_with_timing['time_difference'].max()), 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 3: Histogram by split
    ax3 = fig.add_subplot(2, 2, 3)
    train_data = tp_with_timing[tp_with_timing['split'] == 'train']['time_difference']
    val_data = tp_with_timing[tp_with_timing['split'] == 'val']['time_difference']
    
    ax3.hist(train_data, bins=15, color=COLORS['secondary'], edgecolor='white', 
             alpha=0.7, label=f'Train (n={len(train_data)})')
    ax3.hist(val_data, bins=15, color=COLORS['tp'], edgecolor='white', 
             alpha=0.7, label=f'Val (n={len(val_data)})')
    ax3.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Time Difference by Split', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Scatter - Time Diff vs Probability
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(tp_with_timing['time_difference'], tp_with_timing['prob'], 
                          c=tp_with_timing['prob'], cmap='Greens', 
                          s=30, alpha=0.6, edgecolors='none')
    
    ax4.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Time Difference vs Model Confidence', fontsize=14, fontweight='bold')
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


def plot_confidence_analysis(tp_df, tn_df, output_dir):
    """Analyze prediction confidence for correct predictions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('PREDICTION CONFIDENCE ANALYSIS\n(Correct Predictions)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram of TP probabilities
    ax1 = axes[0]
    ax1.hist(tp_df['prob'], bins=20, color=COLORS['tp'], edgecolor='white', alpha=0.8)
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax1.axvline(tp_df['prob'].mean(), color='darkgreen', linestyle='-', linewidth=2, 
                label=f'Mean: {tp_df["prob"].mean():.3f}')
    ax1.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title(f'True Positives (n={len(tp_df)})\n(Higher prob = more confident)', 
                  fontsize=12, fontweight='bold', color=COLORS['tp'])
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    
    # Plot 2: Histogram of TN probabilities
    ax2 = axes[1]
    ax2.hist(tn_df['prob'], bins=20, color=COLORS['tn'], edgecolor='white', alpha=0.8)
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax2.axvline(tn_df['prob'].mean(), color='darkblue', linestyle='-', linewidth=2, 
                label=f'Mean: {tn_df["prob"].mean():.3f}')
    ax2.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title(f'True Negatives (n={len(tn_df)})\n(Lower prob = more confident)', 
                  fontsize=12, fontweight='bold', color=COLORS['tn'])
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    
    # Plot 3: Combined box plot
    ax3 = axes[2]
    data = [tp_df['prob'].values, tn_df['prob'].values]
    bp = ax3.boxplot(data, patch_artist=True, labels=['True Pos\n(Detected)', 'True Neg\n(Safe)'])
    bp['boxes'][0].set_facecolor(COLORS['tp'])
    bp['boxes'][1].set_facecolor(COLORS['tn'])
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


def save_summary(tp_df, tn_df, output_dir):
    """Save summary statistics to text file."""
    
    tp_with_timing = tp_df.dropna(subset=['time_of_event', 'time_of_alert'])
    
    summary = []
    summary.append("=" * 70)
    summary.append("CORRECT PREDICTIONS ANALYSIS SUMMARY")
    summary.append("=" * 70)
    
    summary.append(f"\nTotal correct predictions: {len(tp_df) + len(tn_df)}")
    summary.append(f"  - True Positives: {len(tp_df)} (correctly detected accidents)")
    summary.append(f"  - True Negatives: {len(tn_df)} (correctly identified safe videos)")
    
    summary.append("\n" + "-" * 70)
    summary.append("TRUE POSITIVES (Correctly Detected Accidents)")
    summary.append("-" * 70)
    
    summary.append(f"\nBy split:")
    summary.append(f"  - Training: {len(tp_df[tp_df['split'] == 'train'])}")
    summary.append(f"  - Validation: {len(tp_df[tp_df['split'] == 'val'])}")
    
    summary.append(f"\nPrediction confidence:")
    summary.append(f"  - Mean probability: {tp_df['prob'].mean():.4f}")
    summary.append(f"  - Std: {tp_df['prob'].std():.4f}")
    summary.append(f"  - Min: {tp_df['prob'].min():.4f}")
    summary.append(f"  - Max: {tp_df['prob'].max():.4f}")
    
    if len(tp_with_timing) > 0:
        summary.append(f"\nTime difference (event - alert) for {len(tp_with_timing)} videos:")
        summary.append(f"  - Mean: {tp_with_timing['time_difference'].mean():.3f}s")
        summary.append(f"  - Std: {tp_with_timing['time_difference'].std():.3f}s")
        summary.append(f"  - Min: {tp_with_timing['time_difference'].min():.3f}s")
        summary.append(f"  - Max: {tp_with_timing['time_difference'].max():.3f}s")
        summary.append(f"  - Median: {tp_with_timing['time_difference'].median():.3f}s")
    
    summary.append("\n" + "-" * 70)
    summary.append("TRUE NEGATIVES (Correctly Identified Safe Videos)")
    summary.append("-" * 70)
    
    summary.append(f"\nBy split:")
    summary.append(f"  - Training: {len(tn_df[tn_df['split'] == 'train'])}")
    summary.append(f"  - Validation: {len(tn_df[tn_df['split'] == 'val'])}")
    
    summary.append(f"\nPrediction confidence:")
    summary.append(f"  - Mean probability: {tn_df['prob'].mean():.4f}")
    summary.append(f"  - Std: {tn_df['prob'].std():.4f}")
    summary.append(f"  - Min: {tn_df['prob'].min():.4f}")
    summary.append(f"  - Max: {tn_df['prob'].max():.4f}")
    
    summary.append("\n" + "=" * 70)
    
    # Save
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"  ✅ Saved: summary_statistics.txt")
    
    # Also print
    print('\n'.join(summary))


def main():
    print("=" * 70)
    print("ANALYZING CORRECT PREDICTIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    tp_df, tn_df = load_data()
    print(f"  - True positives: {len(tp_df)}")
    print(f"  - True negatives: {len(tn_df)}")
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_true_positives_analysis(tp_df, OUTPUT_DIR)
    plot_confidence_analysis(tp_df, tn_df, OUTPUT_DIR)
    
    # Save summary
    print("\nGenerating summary...")
    save_summary(tp_df, tn_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
