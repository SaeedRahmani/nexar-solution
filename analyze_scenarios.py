"""
Analyze model performance by scenario type.

This script:
1. Loads scenario annotations from two annotation rounds
2. Matches annotations between rounds (Level 1 and Level 2 agreement)
3. Merges predictions with annotations
4. Calculates per-scenario accuracy metrics

Usage:
    python analyze_scenarios.py

Outputs to scenario_analysis/:
    - Level_1_matched.csv, Level_2_matched.csv: Matched annotations
    - round_1_prediction.csv, round_2_prediction.csv: Predictions with annotations
    - *_level1_accuracy.csv, *_level2_accuracy.csv: Per-scenario accuracy
"""
import pandas as pd
import os


def load_annotations(round1_path, round2_path):
    """Load both annotation rounds"""
    r1 = pd.read_csv(round1_path)
    r2 = pd.read_csv(round2_path)
    
    # Normalize video names - remove .mp4 extension and convert to int for matching
    # Video names are like "00022.mp4", need to convert to 22
    r1['video_id'] = r1['Video Name'].str.replace('.mp4', '', regex=False).astype(int)
    r2['video_id'] = r2['Video Name'].str.replace('.mp4', '', regex=False).astype(int)
    
    print(f"Round 1: {len(r1)} annotations")
    print(f"Round 2: {len(r2)} annotations")
    
    return r1, r2


def match_annotations(r1, r2, output_dir):
    """Match annotations between rounds"""
    # Merge on video_id
    merged = r1.merge(
        r2, 
        on='video_id', 
        suffixes=('_r1', '_r2'),
        how='inner'
    )
    
    print(f"\n{len(merged)} videos with annotations from both rounds")
    
    # Level 1 matches (only Level 1 agrees)
    level1_match = merged[merged['Scenario Level 1_r1'] == merged['Scenario Level 1_r2']].copy()
    print(f"Level 1 agreement: {len(level1_match)}/{len(merged)} = {100*len(level1_match)/len(merged):.1f}%")
    
    # Level 2 matches (Level 1 must already match, then Level 2 also matches)
    # This is a SUBSET of level1_match
    level2_match = level1_match[
        level1_match['Scenario Level 2_r1'] == level1_match['Scenario Level 2_r2']
    ].copy()
    print(f"Level 2 agreement (among Level 1 matched): {len(level2_match)}/{len(level1_match)} = {100*len(level2_match)/len(level1_match):.1f}%")
    
    # Save matched annotations
    level1_match_simple = level1_match[[
        'video_id', 'Video Name_r1', 
        'Scenario Level 1_r1', 
        'Scenario Level 2_r1', 'Scenario Level 2_r2'
    ]].copy()
    level1_match_simple.columns = ['video_id', 'video_name', 'scenario_level_1', 'scenario_level_2_r1', 'scenario_level_2_r2']
    
    # Level 2 matched: include both Level 1 and Level 2 columns
    level2_match_simple = level2_match[[
        'video_id', 'Video Name_r1',
        'Scenario Level 1_r1', 'Scenario Level 2_r1'
    ]].copy()
    level2_match_simple.columns = ['video_id', 'video_name', 'scenario_level_1', 'scenario_level_2']
    
    # Save
    level1_path = os.path.join(output_dir, 'Level_1_matched.csv')
    level2_path = os.path.join(output_dir, 'Level_2_matched.csv')
    
    level1_match_simple.to_csv(level1_path, index=False)
    level2_match_simple.to_csv(level2_path, index=False)
    
    print(f"\nSaved:\n  {level1_path}\n  {level2_path}")
    
    return level1_match_simple, level2_match_simple


def merge_with_predictions(predictions_path, annotations_df, output_path, annotation_source):
    """Merge predictions with annotations"""
    # Load predictions
    predictions = pd.read_csv(predictions_path)
    print(f"\n{len(predictions)} videos with predictions")
    
    # Merge
    merged = annotations_df.merge(
        predictions[['video_id', 'pred', 'prob', 'true_label']], 
        on='video_id',
        how='inner'
    )
    
    print(f"{len(merged)} videos matched with {annotation_source} annotations")
    
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


def main():
    """Main execution"""
    # Paths
    round1_path = 'scenario_labels/round_1.csv'
    round2_path = 'scenario_labels/round_2.csv'
    predictions_path = 'aggregated_results/aggregated_predictions_any.csv'
    output_dir = 'scenario_analysis'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("STEP 1: Load and match annotations")
    print("="*80)
    
    # Load annotations
    r1, r2 = load_annotations(round1_path, round2_path)
    
    # Match annotations
    level1_matched, level2_matched = match_annotations(r1, r2, output_dir)
    
    print("\n" + "="*80)
    print("STEP 2: Merge with predictions")
    print("="*80)
    
    # Merge round 1 with predictions
    r1_pred = merge_with_predictions(
        predictions_path,
        r1[['video_id', 'Video Name', 'Scenario Level 1', 'Scenario Level 2']].rename(columns={
            'Video Name': 'video_name',
            'Scenario Level 1': 'scenario_level_1',
            'Scenario Level 2': 'scenario_level_2'
        }),
        os.path.join(output_dir, 'round_1_prediction.csv'),
        'round 1'
    )
    
    # Merge round 2 with predictions
    r2_pred = merge_with_predictions(
        predictions_path,
        r2[['video_id', 'Video Name', 'Scenario Level 1', 'Scenario Level 2']].rename(columns={
            'Video Name': 'video_name',
            'Scenario Level 1': 'scenario_level_1',
            'Scenario Level 2': 'scenario_level_2'
        }),
        os.path.join(output_dir, 'round_2_prediction.csv'),
        'round 2'
    )
    
    # Merge Level 1 matched with predictions
    level1_pred = merge_with_predictions(
        predictions_path,
        level1_matched,
        os.path.join(output_dir, 'level_1_matched_prediction.csv'),
        'Level 1 matched'
    )
    
    # Merge Level 2 matched with predictions
    level2_pred = merge_with_predictions(
        predictions_path,
        level2_matched,
        os.path.join(output_dir, 'level_2_matched_prediction.csv'),
        'Level 2 matched'
    )
    
    print("\n" + "="*80)
    print("STEP 3: Calculate per-scenario accuracy")
    print("="*80)
    
    # Calculate accuracy for each dataset
    datasets = [
        (r1_pred, 'round_1', 'Round 1', True),
        (r2_pred, 'round_2', 'Round 2', True),
        (level1_pred, 'level_1_matched', 'Level 1 Matched', False),  # Skip Level 2 - has both r1 and r2 columns
        (level2_pred, 'level_2_matched', 'Level 2 Matched', True)
    ]
    
    for df, prefix, name, do_level2 in datasets:
        print(f"\n{'='*80}")
        print(f"{name} - Overall: {len(df)} videos")
        overall_acc = 100 * (df['predicted_label'] == df['true_label']).sum() / len(df)
        print(f"Overall Accuracy: {overall_acc:.1f}%")
        
        # Level 1 accuracy
        calculate_scenario_accuracy(
            df, 
            'scenario_level_1',
            os.path.join(output_dir, f'{prefix}_level1_accuracy.csv')
        )
        
        # Level 2 accuracy (with Level 1 context)
        if do_level2:
            calculate_scenario_accuracy(
                df,
                ['scenario_level_1', 'scenario_level_2'],
                os.path.join(output_dir, f'{prefix}_level2_accuracy.csv')
            )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
