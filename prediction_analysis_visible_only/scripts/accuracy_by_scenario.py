"""
Accuracy by Scenario Analysis - VISIBLE ONLY SUBSET

This script calculates accuracy broken down by scenario type (Level 1 and Level 2)
for the visible only subset (excluding "Target not visible" videos).

Usage:
    python prediction_analysis_visible_only/scripts/accuracy_by_scenario.py [--dataset VAL|ALL]

Outputs to prediction_analysis_visible_only/accuracy_per_scenario_{dataset}/:
    - accuracy_level1.png/pdf
    - accuracy_level2.png/pdf
    - confusion_matrix.png/pdf
    - accuracy_report.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Colors
COLORS = {
    'correct': '#4CAF50',
    'wrong': '#E63946',
    'primary': '#2E86AB',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def load_data(dataset='ALL'):
    """Load predictions and filtered labels."""
    
    base_dir = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only")
    predictions_path = os.path.join(PROJECT_ROOT, f"video_predictions_{dataset}/aggregated_predictions_any.csv")
    labels_path = os.path.join(base_dir, "filtered_labels/final_matched_visible_only.csv")
    
    print(f"Loading data for {dataset}...")
    
    # Load predictions
    predictions = pd.read_csv(predictions_path)
    predictions['video_id'] = predictions['video_id'].astype(str).str.zfill(5)
    print(f"  Predictions: {len(predictions)} videos")
    
    # Load filtered labels
    labels = pd.read_csv(labels_path)
    labels['video_id'] = labels['Video Name'].str.replace('.mp4', '', regex=False)
    labels = labels.rename(columns={
        'Scenario Level 1': 'scenario_level_1',
        'Scenario Level 2': 'scenario_level_2',
    })
    print(f"  Filtered labels (visible only): {len(labels)} videos")
    
    # Merge
    merged = predictions.merge(
        labels[['video_id', 'scenario_level_1', 'scenario_level_2']],
        on='video_id',
        how='inner'
    )
    print(f"  Merged: {len(merged)} videos")
    
    # Add outcome
    merged['is_correct'] = merged['pred'] == merged['true_label']
    merged['outcome'] = 'Unknown'
    merged.loc[(merged['pred'] == 1) & (merged['true_label'] == 1), 'outcome'] = 'TP'
    merged.loc[(merged['pred'] == 0) & (merged['true_label'] == 0), 'outcome'] = 'TN'
    merged.loc[(merged['pred'] == 1) & (merged['true_label'] == 0), 'outcome'] = 'FP'
    merged.loc[(merged['pred'] == 0) & (merged['true_label'] == 1), 'outcome'] = 'FN'
    
    return merged


def plot_accuracy_by_scenario(df, level_col, title_suffix, output_name, output_dir, dataset):
    """Plot accuracy bar chart by scenario."""
    
    # Calculate accuracy per scenario
    accuracy_stats = df.groupby(level_col).agg({
        'is_correct': ['sum', 'count', 'mean']
    }).reset_index()
    accuracy_stats.columns = [level_col, 'correct', 'total', 'accuracy']
    accuracy_stats['accuracy_pct'] = accuracy_stats['accuracy'] * 100
    accuracy_stats = accuracy_stats.sort_values('accuracy_pct', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(accuracy_stats) * 0.5)))
    
    # Color bars by accuracy
    colors = plt.cm.RdYlGn(accuracy_stats['accuracy'])
    
    bars = ax.barh(range(len(accuracy_stats)), accuracy_stats['accuracy_pct'], 
                   color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(accuracy_stats)))
    ax.set_yticklabels(accuracy_stats[level_col], fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Accuracy by Scenario - {title_suffix}\n[VISIBLE ONLY - {dataset}]', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.axvline(x=df['is_correct'].mean() * 100, color='black', linestyle='--', 
               linewidth=2, label=f'Overall: {df["is_correct"].mean()*100:.1f}%')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add labels
    for i, (bar, acc, total) in enumerate(zip(bars, accuracy_stats['accuracy_pct'], accuracy_stats['total'])):
        ax.text(acc + 1, i, f'{acc:.1f}% (n={total})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return accuracy_stats


def plot_confusion_matrix(df, output_dir, dataset):
    """Plot confusion matrix."""
    
    y_true = df['true_label']
    y_pred = df['pred']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: No Accident', 'Predicted: Accident'], fontsize=11)
    ax.set_yticklabels(['Actual: No Accident', 'Actual: Accident'], fontsize=11)
    
    # Add values
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() / 2 else 'black'
            label = ['TN', 'FP', 'FN', 'TP'][i * 2 + j]
            ax.text(j, i, f'{label}\n{val}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=color)
    
    ax.set_title(f'Confusion Matrix\n[VISIBLE ONLY - {dataset}]', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: confusion_matrix.png/pdf")
    return cm


def save_accuracy_report(df, output_dir, dataset):
    """Save detailed accuracy report."""
    
    with open(os.path.join(output_dir, 'accuracy_report.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ACCURACY BY SCENARIO REPORT - VISIBLE ONLY - {dataset}\n")
        f.write("(Excluding videos where 'Target not visible' is True in ANY round)\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        y_true = df['true_label']
        y_pred = df['pred']
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total videos: {len(df)}\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%\n\n")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        f.write("Confusion Matrix:\n")
        f.write(f"  TN={cm[0,0]}, FP={cm[0,1]}\n")
        f.write(f"  FN={cm[1,0]}, TP={cm[1,1]}\n\n")
        
        # Classification report
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['No Accident', 'Accident']))
        f.write("\n")
        
        # Level 1 breakdown
        f.write("=" * 80 + "\n")
        f.write("LEVEL 1 SCENARIO BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        level1_stats = df.groupby('scenario_level_1').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'outcome': lambda x: dict(x.value_counts())
        }).reset_index()
        level1_stats.columns = ['Scenario', 'Total', 'Correct', 'Accuracy', 'Outcomes']
        level1_stats = level1_stats.sort_values('Accuracy', ascending=False)
        
        f.write(f"{'Scenario':<30} {'Total':>8} {'Acc%':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}\n")
        f.write("-" * 80 + "\n")
        for _, row in level1_stats.iterrows():
            outcomes = row['Outcomes']
            tp = outcomes.get('TP', 0)
            tn = outcomes.get('TN', 0)
            fp = outcomes.get('FP', 0)
            fn = outcomes.get('FN', 0)
            f.write(f"{str(row['Scenario'])[:30]:<30} {row['Total']:>8.0f} {row['Accuracy']*100:>7.1f}% "
                    f"{tp:>6} {tn:>6} {fp:>6} {fn:>6}\n")
        
        # Level 2 breakdown
        f.write("\n" + "=" * 80 + "\n")
        f.write("LEVEL 2 SCENARIO BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        level2_stats = df.groupby('scenario_level_2').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'outcome': lambda x: dict(x.value_counts())
        }).reset_index()
        level2_stats.columns = ['Scenario', 'Total', 'Correct', 'Accuracy', 'Outcomes']
        level2_stats = level2_stats.sort_values('Accuracy', ascending=False)
        
        f.write(f"{'Scenario':<40} {'Total':>8} {'Acc%':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}\n")
        f.write("-" * 90 + "\n")
        for _, row in level2_stats.iterrows():
            outcomes = row['Outcomes']
            tp = outcomes.get('TP', 0)
            tn = outcomes.get('TN', 0)
            fp = outcomes.get('FP', 0)
            fn = outcomes.get('FN', 0)
            f.write(f"{str(row['Scenario'])[:40]:<40} {row['Total']:>8.0f} {row['Accuracy']*100:>7.1f}% "
                    f"{tp:>6} {tn:>6} {fp:>6} {fn:>6}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✅ Saved: accuracy_report.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ALL', choices=['VAL', 'ALL'],
                        help='Dataset to analyze: VAL or ALL')
    args = parser.parse_args()
    
    dataset = args.dataset
    
    print("=" * 70)
    print(f"ACCURACY BY SCENARIO - VISIBLE ONLY - {dataset}")
    print("=" * 70)
    
    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, f"prediction_analysis_visible_only/accuracy_per_scenario_{dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(dataset)
    
    # Generate plots
    print("\nGenerating accuracy plots...")
    plot_accuracy_by_scenario(df, 'scenario_level_1', 'Level 1', 
                              'accuracy_level1', output_dir, dataset)
    plot_accuracy_by_scenario(df, 'scenario_level_2', 'Level 2', 
                              'accuracy_level2', output_dir, dataset)
    
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(df, output_dir, dataset)
    
    print("\nSaving accuracy report...")
    save_accuracy_report(df, output_dir, dataset)
    
    print(f"\n✅ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
