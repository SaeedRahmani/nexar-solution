"""
Comparative Analysis: Wrong vs Correct Predictions - VISIBLE ONLY SUBSET

This script compares characteristics of wrong predictions vs correct predictions
for the visible only subset (excluding "Target not visible" videos).

Usage:
    python prediction_analysis_visible_only/scripts/comparative_analysis.py

Outputs to prediction_analysis_visible_only/comparative_analysis/:
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
BASE_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only")
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
}


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0


def main():
    print("=" * 70)
    print("COMPARATIVE ANALYSIS - VISIBLE ONLY SUBSET")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(WRONG_CSV) or not os.path.exists(CORRECT_CSV):
        print("⚠️  Required files not found. Run extract_predictions.py first.")
        return
    
    wrong_df = pd.read_csv(WRONG_CSV)
    correct_df = pd.read_csv(CORRECT_CSV)
    
    print(f"\nLoaded data:")
    print(f"  Wrong predictions: {len(wrong_df)}")
    print(f"  Correct predictions: {len(correct_df)}")
    
    # Filter to only positive labels (accidents) for time difference comparison
    wrong_positive = wrong_df[wrong_df['true_label'] == 1]  # FN
    correct_positive = correct_df[correct_df['true_label'] == 1]  # TP
    
    # Time difference comparison
    if 'time_diff' in wrong_df.columns:
        wrong_time = wrong_positive[wrong_positive['time_diff'].notna() & (wrong_positive['time_diff'] > 0)]['time_diff']
        correct_time = correct_positive[correct_positive['time_diff'].notna() & (correct_positive['time_diff'] > 0)]['time_diff']
        
        if len(wrong_time) > 0 and len(correct_time) > 0:
            # Histogram comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle('Time Difference: Wrong vs Correct Predictions\n[VISIBLE ONLY SUBSET]', 
                         fontsize=14, fontweight='bold')
            
            bins = np.linspace(0, max(wrong_time.max(), correct_time.max()), 25)
            ax.hist(wrong_time, bins=bins, alpha=0.7, label=f'Wrong (FN) n={len(wrong_time)}', 
                    color=COLORS['wrong'], edgecolor='white')
            ax.hist(correct_time, bins=bins, alpha=0.7, label=f'Correct (TP) n={len(correct_time)}', 
                    color=COLORS['correct'], edgecolor='white')
            ax.axvline(wrong_time.mean(), color=COLORS['wrong'], linestyle='--', linewidth=2)
            ax.axvline(correct_time.mean(), color=COLORS['correct'], linestyle='--', linewidth=2)
            ax.set_xlabel('Time Difference (seconds)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_histograms.png'), 
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_histograms.pdf'), 
                        bbox_inches='tight')
            plt.close()
            print("\n✅ Saved: time_diff_comparison_histograms.png/pdf")
            
            # Box plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot([wrong_time, correct_time], patch_artist=True)
            bp['boxes'][0].set_facecolor(COLORS['wrong'])
            bp['boxes'][1].set_facecolor(COLORS['correct'])
            ax.set_xticklabels(['Wrong (FN)', 'Correct (TP)'])
            ax.set_ylabel('Time Difference (seconds)', fontsize=12)
            ax.set_title('Time Difference Comparison\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_boxplots.png'), 
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_boxplots.pdf'), 
                        bbox_inches='tight')
            plt.close()
            print("✅ Saved: time_diff_comparison_boxplots.png/pdf")
            
            # Violin plot
            fig, ax = plt.subplots(figsize=(10, 6))
            parts = ax.violinplot([wrong_time, correct_time], positions=[1, 2], showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(COLORS['wrong'])
            parts['bodies'][1].set_facecolor(COLORS['correct'])
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Wrong (FN)', 'Correct (TP)'])
            ax.set_ylabel('Time Difference (seconds)', fontsize=12)
            ax.set_title('Time Difference Distribution\n[VISIBLE ONLY SUBSET]', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_violin.png'), 
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_diff_comparison_violin.pdf'), 
                        bbox_inches='tight')
            plt.close()
            print("✅ Saved: time_diff_comparison_violin.png/pdf")
            
            # Statistical tests
            u_stat, p_value = stats.mannwhitneyu(wrong_time, correct_time, alternative='two-sided')
            effect_size = cohens_d(wrong_time, correct_time)
            
            # Save statistics
            with open(os.path.join(OUTPUT_DIR, 'statistical_comparison.txt'), 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("STATISTICAL COMPARISON - VISIBLE ONLY SUBSET\n")
                f.write("(Excluding videos where 'Target not visible' is True)\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("TIME DIFFERENCE COMPARISON (Accidents only: FN vs TP)\n")
                f.write("-" * 50 + "\n")
                f.write(f"\nWrong Predictions (FN - Missed Accidents):\n")
                f.write(f"  n = {len(wrong_time)}\n")
                f.write(f"  Mean: {wrong_time.mean():.3f}s\n")
                f.write(f"  Median: {wrong_time.median():.3f}s\n")
                f.write(f"  Std: {wrong_time.std():.3f}s\n")
                
                f.write(f"\nCorrect Predictions (TP - Detected Accidents):\n")
                f.write(f"  n = {len(correct_time)}\n")
                f.write(f"  Mean: {correct_time.mean():.3f}s\n")
                f.write(f"  Median: {correct_time.median():.3f}s\n")
                f.write(f"  Std: {correct_time.std():.3f}s\n")
                
                f.write(f"\nStatistical Test (Mann-Whitney U):\n")
                f.write(f"  U-statistic: {u_stat:.2f}\n")
                f.write(f"  p-value: {p_value:.4f}\n")
                f.write(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}\n")
                f.write(f"\nEffect Size (Cohen's d): {effect_size:.3f}\n")
                
                if abs(effect_size) < 0.2:
                    effect_interpretation = "negligible"
                elif abs(effect_size) < 0.5:
                    effect_interpretation = "small"
                elif abs(effect_size) < 0.8:
                    effect_interpretation = "medium"
                else:
                    effect_interpretation = "large"
                f.write(f"  Interpretation: {effect_interpretation}\n")
            
            print("✅ Saved: statistical_comparison.txt")
    
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
