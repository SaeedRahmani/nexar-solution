"""
Analyze Correct Predictions - VISIBLE ONLY SUBSET

This script analyzes true positives (correctly predicted accidents) and 
true negatives (correctly predicted non-accidents) for the visible only subset.

Usage:
    python prediction_analysis_visible_only/scripts/analyze_correct_predictions.py

Outputs to prediction_analysis_visible_only/correct_predictions/:
    - time_difference_analysis.png/pdf
    - prediction_confidence_analysis.png/pdf
    - summary_statistics.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only/correct_predictions")
CORRECT_CSV = os.path.join(OUTPUT_DIR, "all_correct_predictions.csv")
TP_CSV = os.path.join(OUTPUT_DIR, "true_positives.csv")
TN_CSV = os.path.join(OUTPUT_DIR, "true_negatives.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'tp': '#2E86AB',
    'tn': '#A23B72',
    'correct': '#4CAF50',
}


def main():
    print("=" * 70)
    print("ANALYZE CORRECT PREDICTIONS - VISIBLE ONLY SUBSET")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(CORRECT_CSV):
        print("⚠️  Correct predictions file not found. Run extract_predictions.py first.")
        return
    
    correct_df = pd.read_csv(CORRECT_CSV)
    tp_df = pd.read_csv(TP_CSV) if os.path.exists(TP_CSV) else pd.DataFrame()
    tn_df = pd.read_csv(TN_CSV) if os.path.exists(TN_CSV) else pd.DataFrame()
    
    print(f"\nLoaded data:")
    print(f"  Total correct predictions: {len(correct_df)}")
    print(f"  True Positives (TP): {len(tp_df)}")
    print(f"  True Negatives (TN): {len(tn_df)}")
    
    # Time difference analysis for TP (correctly predicted accidents)
    if len(tp_df) > 0 and 'time_diff' in tp_df.columns:
        tp_with_time = tp_df[tp_df['time_diff'].notna() & (tp_df['time_diff'] > 0)]
        
        if len(tp_with_time) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Time Difference Analysis - Correct Accident Predictions (TP)\n[VISIBLE ONLY SUBSET]', 
                         fontsize=14, fontweight='bold')
            
            # Histogram
            ax1 = axes[0]
            ax1.hist(tp_with_time['time_diff'], bins=20, color=COLORS['tp'], 
                     edgecolor='white', alpha=0.8)
            ax1.axvline(tp_with_time['time_diff'].mean(), color='black', 
                        linestyle='--', linewidth=2, label=f"Mean: {tp_with_time['time_diff'].mean():.2f}s")
            ax1.axvline(tp_with_time['time_diff'].median(), color='gray', 
                        linestyle=':', linewidth=2, label=f"Median: {tp_with_time['time_diff'].median():.2f}s")
            ax1.set_xlabel('Time Difference (seconds)', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Distribution of Time Differences', fontsize=12)
            ax1.legend()
            
            # Box plot
            ax2 = axes[1]
            bp = ax2.boxplot([tp_with_time['time_diff']], patch_artist=True)
            bp['boxes'][0].set_facecolor(COLORS['tp'])
            ax2.set_ylabel('Time Difference (seconds)', fontsize=12)
            ax2.set_xticklabels(['Correct Predictions'])
            ax2.set_title('Time Difference Box Plot', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_difference_analysis.png'), 
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'time_difference_analysis.pdf'), 
                        bbox_inches='tight')
            plt.close()
            print("\n✅ Saved: time_difference_analysis.png/pdf")
    
    # Confidence analysis
    if 'confidence' in correct_df.columns or 'prob' in correct_df.columns:
        conf_col = 'confidence' if 'confidence' in correct_df.columns else 'prob'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Prediction Confidence Analysis - Correct Predictions\n[VISIBLE ONLY SUBSET]', 
                     fontsize=14, fontweight='bold')
        
        # Histogram by type
        ax1 = axes[0]
        if len(tp_df) > 0 and conf_col in tp_df.columns:
            ax1.hist(tp_df[conf_col], bins=20, alpha=0.7, label='True Positives', 
                     color=COLORS['tp'], edgecolor='white')
        if len(tn_df) > 0 and conf_col in tn_df.columns:
            ax1.hist(tn_df[conf_col], bins=20, alpha=0.7, label='True Negatives', 
                     color=COLORS['tn'], edgecolor='white')
        ax1.set_xlabel('Confidence/Probability', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Confidence Distribution by Prediction Type', fontsize=12)
        ax1.legend()
        
        # Box plot comparison
        ax2 = axes[1]
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []
        
        if len(tp_df) > 0 and conf_col in tp_df.columns:
            data_to_plot.append(tp_df[conf_col].dropna())
            labels_to_plot.append('TP')
            colors_to_plot.append(COLORS['tp'])
        if len(tn_df) > 0 and conf_col in tn_df.columns:
            data_to_plot.append(tn_df[conf_col].dropna())
            labels_to_plot.append('TN')
            colors_to_plot.append(COLORS['tn'])
        
        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_to_plot):
                patch.set_facecolor(color)
            ax2.set_xticklabels(labels_to_plot)
            ax2.set_ylabel('Confidence/Probability', fontsize=12)
            ax2.set_title('Confidence by Prediction Type', fontsize=12)
        
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
        f.write("CORRECT PREDICTIONS ANALYSIS - VISIBLE ONLY SUBSET\n")
        f.write("(Excluding videos where 'Target not visible' is True)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total correct predictions: {len(correct_df)}\n")
        f.write(f"  True Positives (TP): {len(tp_df)}\n")
        f.write(f"  True Negatives (TN): {len(tn_df)}\n\n")
        
        if len(tp_df) > 0 and 'time_diff' in tp_df.columns:
            tp_with_time = tp_df[tp_df['time_diff'].notna() & (tp_df['time_diff'] > 0)]
            if len(tp_with_time) > 0:
                f.write("Time Difference Statistics (TP only):\n")
                f.write(f"  Mean: {tp_with_time['time_diff'].mean():.3f}s\n")
                f.write(f"  Median: {tp_with_time['time_diff'].median():.3f}s\n")
                f.write(f"  Std: {tp_with_time['time_diff'].std():.3f}s\n")
                f.write(f"  Min: {tp_with_time['time_diff'].min():.3f}s\n")
                f.write(f"  Max: {tp_with_time['time_diff'].max():.3f}s\n")
    
    print("✅ Saved: summary_statistics.txt")
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
