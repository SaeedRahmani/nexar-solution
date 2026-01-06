"""Calculate model accuracy by scenario type - FULL DATASET (ALL).

⚠️  BIASED EVALUATION: This analyzes the ENTIRE dataset including training data.
    - Accuracy will be INFLATED/BIASED (model was trained on this data)
    - Use this ONLY for: Better per-scenario statistics with larger samples
    - For unbiased evaluation use: accuracy_per_scenario_VAL/accuracy_by_scenario.py

This script:
1. Loads final matched labels (from scenario_labels/final_matched_all.csv)
2. Merges with FULL DATASET predictions (train + val)
3. Calculates per-scenario accuracy metrics (BIASED - larger sample sizes)
4. Generates visualization plots

Usage:
    python accuracy_per_scenario_ALL/accuracy_by_scenario.py

Outputs to accuracy_per_scenario_ALL/:
    - final_predictions.csv: Predictions merged with final labels
    - final_level1_accuracy.csv: Level 1 accuracy breakdown
    - final_level2_accuracy.csv: Level 2 accuracy breakdown  
    - final_level1_accuracy.png: Level 1 visualization
    - final_level2_accuracy.png: Level 2 visualization
"""
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_final_labels(final_labels_path):
    """Load final matched labels"""
    df = pd.read_csv(final_labels_path)
    
    # Normalize video names - remove .mp4 extension and convert to int for matching
    df['video_id'] = df['Video Name'].str.replace('.mp4', '', regex=False).astype(int)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Video Name': 'video_name',
        'Scenario Level 1': 'scenario_level_1',
        'Scenario Level 2': 'scenario_level_2',
        'Source': 'label_source'
    })
    
    print(f"Loaded {len(df)} videos with final labels")
    print(f"  - From R1/R2 match: {len(df[df['label_source'] == 'R1_R2_Match'])}")
    print(f"  - From R3 sanity check: {len(df[df['label_source'] == 'R3_SanityCheck'])}")
    
    return df


def merge_with_predictions(predictions_path, labels_df, output_path):
    """Merge predictions with labels"""
    # Load predictions
    predictions = pd.read_csv(predictions_path)
    print(f"\n{len(predictions)} videos with predictions")
    
    # Merge
    merged = labels_df.merge(
        predictions[['video_id', 'pred', 'prob', 'true_label']], 
        on='video_id',
        how='inner'
    )
    
    print(f"{len(merged)} videos matched with final labels")
    
    # Rename for clarity
    merged = merged.rename(columns={
        'pred': 'predicted_label',
        'prob': 'predicted_probability'
    })
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return merged


def calculate_scenario_accuracy(df, scenario_columns, output_path):
    """Calculate accuracy per scenario
    
    Args:
        df: DataFrame with predictions and labels
        scenario_columns: str or list of str - column name(s) to group by
        output_path: Where to save results
    """
    # Handle single column or multiple columns
    if isinstance(scenario_columns, str):
        scenario_columns = [scenario_columns]
    
    # Calculate accuracy per scenario
    grouped = df.groupby(scenario_columns).apply(
        lambda x: pd.Series({
            'total': len(x),
            'correct': (x['predicted_label'] == x['true_label']).sum(),
            'accuracy': 100 * (x['predicted_label'] == x['true_label']).sum() / len(x),
            'avg_prob': x['predicted_probability'].mean()
        }),
        include_groups=False
    ).reset_index()
    
    # Sort by total count
    grouped = grouped.sort_values('total', ascending=False)
    
    # Save
    grouped.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    # Display
    column_name = ' + '.join(scenario_columns)
    print(f"\n{column_name} Performance:")
    print("="*80)
    for _, row in grouped.iterrows():
        # Format display based on number of grouping columns
        if len(scenario_columns) == 1:
            label = f"{row[scenario_columns[0]]:30s}"
        else:
            label = ' -> '.join([str(row[col]) for col in scenario_columns])
            label = f"{label:50s}"
        print(f"{label}: {row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']:5.1f}%  (avg prob: {row['avg_prob']:.3f})")
    
    return grouped


def plot_scenario_accuracy(accuracy_csv_path, output_plot_path, title):
    """Create a Modern 'Dashboard-Style' Horizontal Bar Chart.
    
    Design Philosophy: "Clean & Informative"
    - Layout: Linear list (easiest to read).
    - Visuals: Minimalist bars with direct labeling.
    - Grouping: Level 2 items are visually grouped under Level 1 headers.
    - Clutter Reduction: No axes, no grids, no legends. Just data.
    """
    df = pd.read_csv(accuracy_csv_path)
    
    # Determine if this is Level 1 or Level 2 (hierarchical)
    is_hierarchical = 'scenario_level_2' in df.columns
    
    # Sort Data
    if is_hierarchical:
        # Sort by Level 1 (alphabetical) then Accuracy (descending)
        df = df.sort_values(['scenario_level_1', 'accuracy'], ascending=[True, True])
    else:
        df = df.sort_values('accuracy', ascending=True)
        
    # Setup Figure
    row_height = 0.5
    header_height = 2
    total_height = (len(df) * row_height) + header_height
    if is_hierarchical:
        num_categories = len(df['scenario_level_1'].unique())
        total_height += num_categories * 0.8
        
    fig, ax = plt.subplots(figsize=(16, total_height), facecolor='white')
    
    # Colors (Red -> Yellow -> Green)
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=50, vmax=105)
    
    # Plotting Loop
    y_pos = 0
    
    if is_hierarchical:
        groups = df.groupby('scenario_level_1', sort=False)
        
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
        for _, row in df.iterrows():
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
    
    # Add Title
    ax.text(0, -1.5, title, fontsize=20, fontweight='bold', color='black')
    ax.text(0, -0.8, "Accuracy % by Scenario (Bar Length & Color)", fontsize=12, color='#7f8c8d')

    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot: {output_plot_path}")


def calculate_source_accuracy(df, output_path):
    """Calculate accuracy by label source (R1_R2_Match vs R3_SanityCheck)"""
    grouped = df.groupby('label_source').apply(
        lambda x: pd.Series({
            'total': len(x),
            'correct': (x['predicted_label'] == x['true_label']).sum(),
            'accuracy': 100 * (x['predicted_label'] == x['true_label']).sum() / len(x),
            'avg_prob': x['predicted_probability'].mean()
        }),
        include_groups=False
    ).reset_index()
    
    grouped.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    print("\nAccuracy by Label Source:")
    print("="*60)
    for _, row in grouped.iterrows():
        print(f"{row['label_source']:20s}: {row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']:5.1f}%")
    
    return grouped


def main():
    """Main execution"""
    # Paths (relative to project root)
    final_labels_path = os.path.join(PROJECT_ROOT, 'scenario_labels/final_matched_all.csv')
    predictions_path = os.path.join(PROJECT_ROOT, 'video_predictions_ALL/aggregated_predictions_any.csv')  # Full dataset - train + val (BIASED)
    output_dir = os.path.join(PROJECT_ROOT, 'accuracy_per_scenario_ALL')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("SCENARIO ANALYSIS - FULL DATASET (ALL) - BIASED")
    print("⚠️  WARNING: Includes training data - accuracy is INFLATED!")
    print("="*80)
    
    # Check if files exist
    if not os.path.exists(final_labels_path):
        print(f"Error: Final labels file not found: {final_labels_path}")
        print("Please run scenario_labels/create_final_labels.py first to generate final labels.")
        return
    
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found: {predictions_path}")
        return
    
    print("\n" + "="*80)
    print("STEP 1: Load final labels")
    print("="*80)
    
    labels_df = load_final_labels(final_labels_path)
    
    print("\n" + "="*80)
    print("STEP 2: Merge with predictions")
    print("="*80)
    
    merged_df = merge_with_predictions(
        predictions_path,
        labels_df,
        os.path.join(output_dir, 'final_predictions.csv')
    )
    
    print("\n" + "="*80)
    print("STEP 3: Overall accuracy")
    print("="*80)
    
    overall_acc = 100 * (merged_df['predicted_label'] == merged_df['true_label']).sum() / len(merged_df)
    print(f"\nOverall Accuracy: {overall_acc:.1f}% ({len(merged_df)} videos)")
    
    # Accuracy by label source
    calculate_source_accuracy(
        merged_df,
        os.path.join(output_dir, 'final_source_accuracy.csv')
    )
    
    print("\n" + "="*80)
    print("STEP 4: Per-scenario accuracy")
    print("="*80)
    
    # Level 1 accuracy
    level1_csv = os.path.join(output_dir, 'final_level1_accuracy.csv')
    calculate_scenario_accuracy(
        merged_df, 
        'scenario_level_1',
        level1_csv
    )
    plot_scenario_accuracy(
        level1_csv,
        os.path.join(output_dir, 'final_level1_accuracy.png'),
        'Final Labels - Level 1'
    )
    
    # Level 2 accuracy (with Level 1 context)
    level2_csv = os.path.join(output_dir, 'final_level2_accuracy.csv')
    calculate_scenario_accuracy(
        merged_df,
        ['scenario_level_1', 'scenario_level_2'],
        level2_csv
    )
    plot_scenario_accuracy(
        level2_csv,
        os.path.join(output_dir, 'final_level2_accuracy.png'),
        'Final Labels - Level 2 (Hierarchical)'
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nOutput files:")
    print(f"  - {output_dir}/final_predictions.csv")
    print(f"  - {output_dir}/final_source_accuracy.csv")
    print(f"  - {output_dir}/final_level1_accuracy.csv")
    print(f"  - {output_dir}/final_level1_accuracy.png")
    print(f"  - {output_dir}/final_level2_accuracy.csv")
    print(f"  - {output_dir}/final_level2_accuracy.png")


if __name__ == '__main__':
    main()
