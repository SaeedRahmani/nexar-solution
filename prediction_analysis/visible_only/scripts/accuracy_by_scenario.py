"""
Accuracy by Scenario Analysis - VISIBLE ONLY SUBSET

This script calculates accuracy broken down by scenario type (Level 1 and Level 2)
for the visible only subset (excluding "Target not visible" videos).

Usage:
    python prediction_analysis/visible_only/scripts/accuracy_by_scenario.py [--dataset VAL|ALL]

Outputs to prediction_analysis/visible_only/accuracy_per_scenario_{dataset}/:
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    
    base_dir = os.path.join(PROJECT_ROOT, "prediction_analysis/visible_only")
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
    """Create a Modern 'Dashboard-Style' Horizontal Bar Chart - matching original format.
    
    Design Philosophy: "Clean & Informative"
    - Layout: Linear list (easiest to read).
    - Visuals: Minimalist bars with direct labeling.
    - Grouping: Level 2 items are visually grouped under Level 1 headers.
    - Clutter Reduction: No axes, no grids, no legends. Just data.
    """
    # Determine if this is Level 1 or Level 2 (hierarchical)
    is_hierarchical = (level_col == 'scenario_level_2')
    
    # Calculate accuracy per scenario from the VISIBLE ONLY data passed in
    if is_hierarchical:
        accuracy_stats = df.groupby(['scenario_level_1', 'scenario_level_2']).agg({
            'is_correct': ['sum', 'count', 'mean']
        }).reset_index()
        accuracy_stats.columns = ['scenario_level_1', 'scenario_level_2', 'correct', 'total', 'accuracy']
        accuracy_stats['accuracy'] = accuracy_stats['accuracy'] * 100
        # Sort by Level 1 (alphabetical) then Accuracy (ascending)
        accuracy_stats = accuracy_stats.sort_values(['scenario_level_1', 'accuracy'], ascending=[True, True])
    else:
        accuracy_stats = df.groupby(level_col).agg({
            'is_correct': ['sum', 'count', 'mean']
        }).reset_index()
        accuracy_stats.columns = [level_col, 'correct', 'total', 'accuracy']
        accuracy_stats['accuracy'] = accuracy_stats['accuracy'] * 100
        accuracy_stats = accuracy_stats.sort_values('accuracy', ascending=True)
    
    # Setup Figure
    row_height = 0.5
    header_height = 2
    total_height = (len(accuracy_stats) * row_height) + header_height
    if is_hierarchical:
        num_categories = len(accuracy_stats['scenario_level_1'].unique())
        total_height += num_categories * 0.8
        
    fig, ax = plt.subplots(figsize=(16, total_height), facecolor='white')
    
    # Colors (Red -> Yellow -> Green)
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=50, vmax=105)
    
    # Plotting Loop
    y_pos = 0
    
    if is_hierarchical:
        groups = accuracy_stats.groupby('scenario_level_1', sort=False)
        
        for name, group in groups:
            # Add Category Header
            y_pos += 1
            ax.text(0, y_pos, name.upper(), fontsize=14, fontweight='bold', color='black', va='center')
            ax.hlines(y_pos - 0.3, 0, 100, color='black', linewidth=1)
            y_pos += 0.8
            
            # Plot items in group
            for _, row in group.iterrows():
                color = cmap(norm(row['accuracy']))
                ax.barh(y_pos, row['accuracy'], height=0.6, color=color, alpha=0.9, align='center', edgecolor='none')
                
                ax.text(-1, y_pos, row['scenario_level_2'], ha='right', va='center', fontsize=11, color='black', fontweight='medium')
                
                bar_text_color = 'black' if (60 <= row['accuracy'] <= 92) else 'white'
                
                if row['accuracy'] < 15:
                    ax.text(row['accuracy'] + 1, y_pos, f"{row['accuracy']:.1f}%", 
                           ha='left', va='center', fontsize=10, fontweight='bold', color='black')
                    ax.text(row['accuracy'] + 12, y_pos, f"(n={int(row['total'])})", 
                           ha='left', va='center', fontsize=10, color='black')
                else:
                    ax.text(row['accuracy'] - 1, y_pos, f"{row['accuracy']:.1f}%", 
                           ha='right', va='center', fontsize=10, fontweight='bold', color=bar_text_color)
                    ax.text(row['accuracy'] + 1, y_pos, f"(n={int(row['total'])})", 
                           ha='left', va='center', fontsize=10, color='black')
                
                y_pos += 0.6
            
            y_pos += 0.5
            
    else:
        for _, row in accuracy_stats.iterrows():
            color = cmap(norm(row['accuracy']))
            ax.barh(y_pos, row['accuracy'], height=0.6, color=color, alpha=0.9, align='center')
            
            ax.text(-1, y_pos, row['scenario_level_1'], ha='right', va='center', fontsize=12, color='black', fontweight='medium')
            
            bar_text_color = 'black' if (60 <= row['accuracy'] <= 92) else 'white'
            
            if row['accuracy'] < 15:
                ax.text(row['accuracy'] + 1, y_pos, f"{row['accuracy']:.1f}%", 
                       ha='left', va='center', fontsize=11, fontweight='bold', color='black')
                ax.text(row['accuracy'] + 12, y_pos, f"(n={int(row['total'])})", 
                       ha='left', va='center', fontsize=11, color='black')
            else:
                ax.text(row['accuracy'] - 1, y_pos, f"{row['accuracy']:.1f}%", 
                       ha='right', va='center', fontsize=11, fontweight='bold', color=bar_text_color)
                ax.text(row['accuracy'] + 1, y_pos, f"(n={int(row['total'])})", 
                       ha='left', va='center', fontsize=11, color='black')
            
            y_pos += 0.8

    # Styling
    ax.set_xlim(-40, 115)
    ax.set_ylim(-2, y_pos)
    ax.invert_yaxis()
    ax.axis('off')
    
    # Add Title with two parts: "Level X – " in black, subtitle in red
    if 'Level 1' in title_suffix:
        level_text = 'Level 1 – '
    elif 'Level 2' in title_suffix:
        level_text = 'Level 2 – '
    else:
        level_text = title_suffix + ' – '
    
    # Subtitle based on dataset
    if dataset == 'VAL':
        subtitle = 'Visible Only – Validation Dataset'
    else:
        subtitle = 'Visible Only – Full Dataset'
    
    # Draw level text in black, get its extent
    txt1 = ax.text(0, -1.5, level_text, fontsize=20, fontweight='bold', color='black',
                   transform=ax.transData)
    
    # Force a draw to get the text extent
    fig.canvas.draw()
    bbox = txt1.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_data = bbox.transformed(ax.transData.inverted())
    
    # Draw subtitle right after level text
    ax.text(bbox_data.x1, -1.5, subtitle, fontsize=20, fontweight='bold', color='#C41E3A')
    ax.text(0, -0.8, "Accuracy % by Scenario (Bar Length & Color)", fontsize=12, color='#7f8c8d')

    plt.savefig(os.path.join(output_dir, f'{output_name}.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{output_name}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}.png/pdf")
    return accuracy_stats


def plot_detection_metrics(df, output_dir, dataset):
    """Plot detection rate metrics for incident videos (visible only).
    
    Note: For visible only analysis, this only includes INCIDENT videos
    (where scenario labels exist). Normal videos are not included because
    they don't have visibility annotations.
    """
    import seaborn as sns
    
    y_true = df['true_label']
    y_pred = df['pred']
    
    # Calculate metrics (all are incident videos, so only TP and FN matter)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    total = tp + fn
    recall = tp / total * 100 if total > 0 else 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a simple bar chart showing detected vs missed
    categories = ['Correctly Detected\n(True Positives)', 'Missed\n(False Negatives)']
    values = [tp, fn]
    colors = ['#2E86AB', '#E63946']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(val)}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Add percentage labels
    ax.text(bars[0].get_x() + bars[0].get_width()/2., bars[0].get_height()/2,
            f'{tp/total*100:.1f}%', ha='center', va='center', fontsize=16, 
            fontweight='bold', color='white')
    ax.text(bars[1].get_x() + bars[1].get_width()/2., bars[1].get_height()/2,
            f'{fn/total*100:.1f}%', ha='center', va='center', fontsize=16, 
            fontweight='bold', color='white')
    
    # Styling
    ax.set_ylabel('Number of Videos', fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Title
    if dataset == 'VAL':
        subtitle = 'Visible Only – Validation Dataset'
    else:
        subtitle = 'Visible Only – Full Dataset'
    
    ax.set_title(f'Incident Detection Rate – {subtitle}\n'
                 f'Total: {total} incident videos | Detection Rate: {recall:.1f}%',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_metrics.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'detection_metrics.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: detection_metrics.png/pdf")
    return {'total': total, 'tp': tp, 'fn': fn, 'recall': recall}


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
    output_dir = os.path.join(PROJECT_ROOT, f"prediction_analysis/visible_only/accuracy_per_scenario_{dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(dataset)
    
    # Generate plots
    print("\nGenerating accuracy plots...")
    plot_accuracy_by_scenario(df, 'scenario_level_1', 'Level 1', 
                              'accuracy_level1', output_dir, dataset)
    plot_accuracy_by_scenario(df, 'scenario_level_2', 'Level 2', 
                              'accuracy_level2', output_dir, dataset)
    
    print("\nGenerating detection metrics...")
    plot_detection_metrics(df, output_dir, dataset)
    
    print("\nSaving accuracy report...")
    save_accuracy_report(df, output_dir, dataset)
    
    print(f"\n✅ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
