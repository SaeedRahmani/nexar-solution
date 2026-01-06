"""
Extract and categorize predictions - VISIBLE ONLY SUBSET

This script extracts predictions and categorizes them into:
- True Positives (TP): Correctly predicted accidents
- True Negatives (TN): Correctly predicted non-accidents  
- False Positives (FP): False alarms (predicted accident but wasn't)
- False Negatives (FN): Missed accidents (was accident but predicted non-accident)

SUBSET: Excludes videos where "Target not visible" is True in ANY round.

Usage:
    python prediction_analysis_visible_only/scripts/extract_predictions.py [--dataset VAL|ALL]

Outputs to prediction_analysis_visible_only/{dataset}/:
    - wrong_predictions/false_negatives.csv
    - wrong_predictions/false_positives.csv
    - wrong_predictions/all_wrong_predictions.csv
    - correct_predictions/true_positives.csv
    - correct_predictions/true_negatives.csv
    - correct_predictions/all_correct_predictions.csv
"""

import pandas as pd
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def main(dataset='ALL'):
    print("=" * 70)
    print(f"EXTRACT PREDICTIONS - VISIBLE ONLY SUBSET - {dataset}")
    print("=" * 70)
    
    # Paths
    predictions_path = os.path.join(PROJECT_ROOT, f"video_predictions_{dataset}/aggregated_predictions_any.csv")
    filtered_labels_path = os.path.join(PROJECT_ROOT, "prediction_analysis_visible_only/filtered_labels/final_matched_visible_only.csv")
    train_csv_path = os.path.join(PROJECT_ROOT, "dataset/train.csv")
    output_base = os.path.join(PROJECT_ROOT, f"prediction_analysis_visible_only/{dataset}")
    
    # Create output directories
    wrong_dir = os.path.join(output_base, "wrong_predictions")
    correct_dir = os.path.join(output_base, "correct_predictions")
    os.makedirs(wrong_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = pd.read_csv(predictions_path)
    predictions['video_id'] = predictions['video_id'].astype(str).str.zfill(5)
    print(f"  Total predictions: {len(predictions)}")
    
    # Load train.csv to get time_diff
    print("\nLoading train.csv for time differences...")
    train_df = pd.read_csv(train_csv_path)
    train_df['video_id'] = train_df['id'].apply(lambda x: f"{x:05d}")
    train_df['time_diff'] = train_df['time_of_event'] - train_df['time_of_alert']
    
    # Merge predictions with time_diff
    predictions = predictions.merge(
        train_df[['video_id', 'time_diff', 'time_of_event', 'time_of_alert']],
        on='video_id',
        how='left'
    )
    print(f"  Videos with time_diff: {predictions['time_diff'].notna().sum()}")
    
    # Load filtered labels (visible only)
    print("\nLoading filtered labels (visible only)...")
    labels = pd.read_csv(filtered_labels_path)
    labels['video_id'] = labels['Video Name'].str.replace('.mp4', '', regex=False)
    visible_video_ids = set(labels['video_id'].tolist())
    print(f"  Visible videos: {len(visible_video_ids)}")
    
    # Filter predictions to only include visible videos
    filtered_predictions = predictions[predictions['video_id'].isin(visible_video_ids)]
    print(f"  Predictions after filtering: {len(filtered_predictions)}")
    
    # Categorize predictions
    df = filtered_predictions.copy()
    df['outcome'] = 'Unknown'
    df.loc[(df['pred'] == 1) & (df['true_label'] == 1), 'outcome'] = 'TP'
    df.loc[(df['pred'] == 0) & (df['true_label'] == 0), 'outcome'] = 'TN'
    df.loc[(df['pred'] == 1) & (df['true_label'] == 0), 'outcome'] = 'FP'
    df.loc[(df['pred'] == 0) & (df['true_label'] == 1), 'outcome'] = 'FN'
    
    # Split into categories
    tp = df[df['outcome'] == 'TP']
    tn = df[df['outcome'] == 'TN']
    fp = df[df['outcome'] == 'FP']
    fn = df[df['outcome'] == 'FN']
    
    wrong = df[df['outcome'].isin(['FP', 'FN'])]
    correct = df[df['outcome'].isin(['TP', 'TN'])]
    
    # Save wrong predictions
    fn.to_csv(os.path.join(wrong_dir, 'false_negatives.csv'), index=False)
    fp.to_csv(os.path.join(wrong_dir, 'false_positives.csv'), index=False)
    wrong.to_csv(os.path.join(wrong_dir, 'all_wrong_predictions.csv'), index=False)
    
    # Save correct predictions
    tp.to_csv(os.path.join(correct_dir, 'true_positives.csv'), index=False)
    tn.to_csv(os.path.join(correct_dir, 'true_negatives.csv'), index=False)
    correct.to_csv(os.path.join(correct_dir, 'all_correct_predictions.csv'), index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY - VISIBLE ONLY SUBSET - {dataset}")
    print("=" * 70)
    print(f"\nTotal videos (visible only): {len(df)}")
    print(f"\nPrediction Outcomes:")
    print(f"  True Positives (TP):  {len(tp):4d} ({len(tp)/len(df)*100:.1f}%)")
    print(f"  True Negatives (TN):  {len(tn):4d} ({len(tn)/len(df)*100:.1f}%)")
    print(f"  False Positives (FP): {len(fp):4d} ({len(fp)/len(df)*100:.1f}%)")
    print(f"  False Negatives (FN): {len(fn):4d} ({len(fn)/len(df)*100:.1f}%)")
    print(f"\nAccuracy: {len(correct)/len(df)*100:.2f}%")
    print(f"\n✅ Saved to: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ALL', choices=['VAL', 'ALL'])
    args = parser.parse_args()
    main(args.dataset)
