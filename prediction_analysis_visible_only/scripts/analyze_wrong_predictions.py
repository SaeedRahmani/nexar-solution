"""
Analyze Wrong Predictions - VISIBLE ONLY SUBSET

This script analyzes false negatives (missed accidents) and false positives (false alarms)
for the subset where "Target not visible" is excluded.

Usage:
    python prediction_analysis_visible_only/scripts/analyze_wrong_predictions.py

Outputs to prediction_analysis_visible_only/wrong_predictions/:
    - time_difference_analysis.png/pdf
    - prediction_confidence_analysis.png/pdf
    - summary_statistics.txt
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only/wrong_predictions")
WRONG_CSV = os.path.join(OUTPUT_DIR, "all_wrong_predictions.csv")
FN_CSV = os.path.join(OUTPUT_DIR, "false_negatives.csv")
FP_CSV = os.path.join(OUTPUT_DIR, "false_positives.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'fn': '#E63946',
    'fp': '#F18F01',
    'wrong': '#C62828',
}


def main():
    print("=" * 70)
    print("ANALYZE WRONG PREDICTIONS - VISIBLE ONLY SUBSET")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(WRONG_CSV):
        print("⚠️  Wrong predictions file not found. Run extract_predictions.py first.")
        return
    
    wrong_df = pd.read_csv(WRONG_CSV)
    fn_df = pd.read_csv(FN_CSV) if os.path.exists(FN_CSV) else pd.DataFrame()
    fp_df = pd.read_csv(FP_CSV) if os.path.exists(FP_CSV) else pd.DataFrame()
    
    print(f"\nLoaded data:")
    print(f"  Total wrong predictions: {len(wrong_df)}")
    print(f"  False Negatives (FN): {len(fn_df)}")
    print(f"  False Positives (FP): {len(fp_df)}")
    
    # Time difference analysis for FN (missed accidents)
    if len(fn_df) > 0 and 'time_diff' in fn_df.columns:
        fn_with_time = fn_df[fn_df['time_diff'].notna() & (fn_df['time_diff'] > 0)]
        
        if len(fn_with_time) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Time Difference Analysis - Missed Accidents (FN)\n[VISIBLE ONLY SUBSET]', 
                         fontsize=14, fontweight='bold')
            
            # Histogram
            ax1 = axes[0]
            ax1.hist(fn_with_time['time_diff'], bins=20, color=COLORS['fn'], 
                     edgecolor='white', alpha=0.8)
            ax1.axvline(fn_with_time['time_diff'].mean(), color='black', 
                        linestyle='--', linewidth=2, label=f"Mean: {fn_with_time['time_diff'].mean():.2f}s")
            ax1.axvline(fn_with_time['time_diff'].median(), color='gray', 
                        linestyle=':', linewidth=2, label=f"Median: {fn_with_time['time_diff'].median():.2f}s")
            ax1.set_xlabel('Time Difference (seconds)', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Distribution of Time Differences', fontsize=12)
            ax1.legend()
            
            # Box plot
            ax2 = axes[1]
            bp = ax2.boxplot([fn_with_time['time_diff']], patch_artist=True)
            bp['boxes'][0].set_facecolor(COLORS['fn'])
            ax2.set_ylabel('Time Difference (seconds)', fontsize=12)
            ax2.set_xticklabels(['Missed Accidents'])
            ax2.set_title('Time Difference Box Plot', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_difference_analysis.png'), 
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_difference_analysis.pdf'), 
                        bbox_inches='tight')
            plt.close()
            print("\n✅ Saved: time_difference_analysis.png/pdf")
    
    # Confidence analysis
    if 'confidence' in wrong_df.columns or 'prob' in wrong_df.columns:
        conf_col = 'confidence' if 'confidence' in wrong_df.columns else 'prob'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Prediction Confidence Analysis - Wrong Predictions\n[VISIBLE ONLY SUBSET]', 
                     fontsize=14, fontweight='bold')
        
        # Histogram by type
        ax1 = axes[0]
        if len(fn_df) > 0 and conf_col in fn_df.columns:
            ax1.hist(fn_df[conf_col], bins=20, alpha=0.7, label='False Negatives', 
                     color=COLORS['fn'], edgecolor='white')
        if len(fp_df) > 0 and conf_col in fp_df.columns:
            ax1.hist(fp_df[conf_col], bins=20, alpha=0.7, label='False Positives', 
                     color=COLORS['fp'], edgecolor='white')
        ax1.set_xlabel('Confidence/Probability', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Confidence Distribution by Error Type', fontsize=12)
        ax1.legend()
        
        # Box plot comparison
        ax2 = axes[1]
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []
        
        if len(fn_df) > 0 and conf_col in fn_df.columns:
            data_to_plot.append(fn_df[conf_col].dropna())
            labels_to_plot.append('FN')
            colors_to_plot.append(COLORS['fn'])
        if len(fp_df) > 0 and conf_col in fp_df.columns:
            data_to_plot.append(fp_df[conf_col].dropna())
            labels_to_plot.append('FP')
            colors_to_plot.append(COLORS['fp'])
        
        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_to_plot):
                patch.set_facecolor(color)
            ax2.set_xticklabels(labels_to_plot)
            ax2.set_ylabel('Confidence/Probability', fontsize=12)
            ax2.set_title('Confidence by Error Type', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_confidence_analysis.png'), 
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_confidence_analysis.pdf'), 
                    bbox_inches='tight')
        plt.close()
        print("✅ Saved: prediction_confidence_analysis.png/pdf")
    
    # Save summary statistics
    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("WRONG PREDICTIONS ANALYSIS - VISIBLE ONLY SUBSET\n")
        f.write("(Excluding videos where 'Target not visible' is True)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total wrong predictions: {len(wrong_df)}\n")
        f.write(f"  False Negatives (FN): {len(fn_df)}\n")
        f.write(f"  False Positives (FP): {len(fp_df)}\n\n")
        
        if len(fn_df) > 0 and 'time_diff' in fn_df.columns:
            fn_with_time = fn_df[fn_df['time_diff'].notna() & (fn_df['time_diff'] > 0)]
            if len(fn_with_time) > 0:
                f.write("Time Difference Statistics (FN only):\n")
                f.write(f"  Mean: {fn_with_time['time_diff'].mean():.3f}s\n")
                f.write(f"  Median: {fn_with_time['time_diff'].median():.3f}s\n")
                f.write(f"  Std: {fn_with_time['time_diff'].std():.3f}s\n")
                f.write(f"  Min: {fn_with_time['time_diff'].min():.3f}s\n")
                f.write(f"  Max: {fn_with_time['time_diff'].max():.3f}s\n")
    
    print("✅ Saved: summary_statistics.txt")
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
