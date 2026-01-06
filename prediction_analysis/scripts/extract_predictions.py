"""
Extract and categorize predictions into correct and wrong predictions.

This script:
1. Loads video-level predictions from video_predictions_ALL/
2. Separates into correct (TP, TN) and wrong (FP, FN) predictions
3. Adds timing information for positive videos
4. Saves organized outputs to respective folders

Usage:
    python prediction_analysis/scripts/extract_predictions.py

Outputs:
    - prediction_analysis/wrong_predictions/
        - false_negatives.csv (positive videos predicted as negative)
        - false_positives.csv (negative videos predicted as positive)
        - all_wrong_predictions.csv (combined)
    - prediction_analysis/correct_predictions/
        - true_positives.csv (positive videos correctly predicted)
        - true_negatives.csv (negative videos correctly predicted)
        - all_correct_predictions.csv (combined)
"""

import pandas as pd
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "video_predictions_ALL/aggregated_predictions_any.csv")
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "dataset/train.csv")
BALANCED_DATASET_PATH = os.path.join(PROJECT_ROOT, "balanced_dataset_2s")

# Output directories
WRONG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/wrong_predictions")
CORRECT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/correct_predictions")

os.makedirs(WRONG_OUTPUT_DIR, exist_ok=True)
os.makedirs(CORRECT_OUTPUT_DIR, exist_ok=True)


def get_train_val_split():
    """Determine which videos are in train vs val split."""
    train_videos = set()
    val_videos = set()
    
    for label_dir in ['positive', 'negative']:
        # Train
        train_path = os.path.join(BALANCED_DATASET_PATH, 'train', label_dir)
        if os.path.exists(train_path):
            for filename in os.listdir(train_path):
                if filename.endswith('.avi'):
                    video_id = int(filename.split('_')[0])
                    train_videos.add(video_id)
        
        # Val
        val_path = os.path.join(BALANCED_DATASET_PATH, 'val', label_dir)
        if os.path.exists(val_path):
            for filename in os.listdir(val_path):
                if filename.endswith('.avi'):
                    video_id = int(filename.split('_')[0])
                    val_videos.add(video_id)
    
    return train_videos, val_videos


def load_timing_data():
    """Load timing information from train.csv."""
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['video_id'] = train_df['id']  # Keep as int for merging
    return train_df[['video_id', 'time_of_event', 'time_of_alert', 'target']]


def categorize_predictions(predictions_df, timing_df, train_videos, val_videos):
    """Categorize predictions into TP, TN, FP, FN with timing info."""
    
    # Merge with timing data
    merged = predictions_df.merge(
        timing_df[['video_id', 'time_of_event', 'time_of_alert']],
        on='video_id',
        how='left'
    )
    
    # Calculate time difference for positive videos
    merged['time_difference'] = merged['time_of_event'] - merged['time_of_alert']
    
    # Add split info
    merged['split'] = merged['video_id'].apply(
        lambda x: 'train' if x in train_videos else ('val' if x in val_videos else 'unknown')
    )
    
    # Format video_id as 5-digit string for display
    merged['video_id_str'] = merged['video_id'].apply(lambda x: f"{x:05d}")
    
    # Categorize predictions
    merged['prediction_type'] = merged.apply(
        lambda row: (
            'true_positive' if row['pred'] == 1 and row['true_label'] == 1 else
            'true_negative' if row['pred'] == 0 and row['true_label'] == 0 else
            'false_positive' if row['pred'] == 1 and row['true_label'] == 0 else
            'false_negative'  # pred == 0 and true_label == 1
        ),
        axis=1
    )
    
    merged['is_correct'] = merged['pred'] == merged['true_label']
    
    return merged


def save_categorized_predictions(df):
    """Save predictions to organized folders."""
    
    # Columns to save
    output_cols = [
        'video_id_str', 'pred', 'prob', 'true_label', 
        'prediction_type', 'split',
        'time_of_event', 'time_of_alert', 'time_difference'
    ]
    
    # Rename video_id_str to video_id for output
    def prepare_output(subset):
        out = subset[output_cols].copy()
        out = out.rename(columns={'video_id_str': 'video_id'})
        return out.sort_values(['split', 'video_id'])
    
    # === WRONG PREDICTIONS ===
    wrong = df[~df['is_correct']]
    
    # False Negatives (most critical - missed dangerous videos)
    fn = wrong[wrong['prediction_type'] == 'false_negative']
    fn_out = prepare_output(fn)
    fn_out.to_csv(os.path.join(WRONG_OUTPUT_DIR, 'false_negatives.csv'), index=False)
    
    # False Positives
    fp = wrong[wrong['prediction_type'] == 'false_positive']
    fp_out = prepare_output(fp)
    fp_out.to_csv(os.path.join(WRONG_OUTPUT_DIR, 'false_positives.csv'), index=False)
    
    # All wrong combined
    wrong_out = prepare_output(wrong)
    wrong_out.to_csv(os.path.join(WRONG_OUTPUT_DIR, 'all_wrong_predictions.csv'), index=False)
    
    # === CORRECT PREDICTIONS ===
    correct = df[df['is_correct']]
    
    # True Positives
    tp = correct[correct['prediction_type'] == 'true_positive']
    tp_out = prepare_output(tp)
    tp_out.to_csv(os.path.join(CORRECT_OUTPUT_DIR, 'true_positives.csv'), index=False)
    
    # True Negatives
    tn = correct[correct['prediction_type'] == 'true_negative']
    tn_out = prepare_output(tn)
    tn_out.to_csv(os.path.join(CORRECT_OUTPUT_DIR, 'true_negatives.csv'), index=False)
    
    # All correct combined
    correct_out = prepare_output(correct)
    correct_out.to_csv(os.path.join(CORRECT_OUTPUT_DIR, 'all_correct_predictions.csv'), index=False)
    
    return {
        'true_positive': len(tp),
        'true_negative': len(tn),
        'false_positive': len(fp),
        'false_negative': len(fn)
    }


def main():
    print("=" * 70)
    print("EXTRACTING AND CATEGORIZING PREDICTIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    timing_df = load_timing_data()
    train_videos, val_videos = get_train_val_split()
    
    print(f"  - Predictions: {len(predictions)} videos")
    print(f"  - Train videos: {len(train_videos)}")
    print(f"  - Val videos: {len(val_videos)}")
    
    # Categorize predictions
    print("\nCategorizing predictions...")
    categorized = categorize_predictions(predictions, timing_df, train_videos, val_videos)
    
    # Save outputs
    print("\nSaving categorized predictions...")
    counts = save_categorized_predictions(categorized)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = sum(counts.values())
    correct = counts['true_positive'] + counts['true_negative']
    wrong = counts['false_positive'] + counts['false_negative']
    
    print(f"\nTotal videos: {total}")
    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    print(f"\n✅ CORRECT PREDICTIONS: {correct} ({100 * correct / total:.1f}%)")
    print(f"   - True Positives:  {counts['true_positive']:4d} (correctly detected accidents)")
    print(f"   - True Negatives:  {counts['true_negative']:4d} (correctly identified safe videos)")
    
    print(f"\n❌ WRONG PREDICTIONS: {wrong} ({100 * wrong / total:.1f}%)")
    print(f"   - False Negatives: {counts['false_negative']:4d} ⚠️  CRITICAL (missed accidents)")
    print(f"   - False Positives: {counts['false_positive']:4d} (false alarms)")
    
    # Split breakdown
    train_df = categorized[categorized['split'] == 'train']
    val_df = categorized[categorized['split'] == 'val']
    
    print(f"\n--- Training Set (BIASED) ---")
    train_correct = train_df['is_correct'].sum()
    print(f"   Accuracy: {100 * train_correct / len(train_df):.2f}% ({train_correct}/{len(train_df)})")
    
    print(f"\n--- Validation Set (UNBIASED) ---")
    val_correct = val_df['is_correct'].sum()
    print(f"   Accuracy: {100 * val_correct / len(val_df):.2f}% ({val_correct}/{len(val_df)})")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\nWrong predictions → {WRONG_OUTPUT_DIR}/")
    print(f"  - false_negatives.csv ({counts['false_negative']} videos)")
    print(f"  - false_positives.csv ({counts['false_positive']} videos)")
    print(f"  - all_wrong_predictions.csv ({wrong} videos)")
    print(f"\nCorrect predictions → {CORRECT_OUTPUT_DIR}/")
    print(f"  - true_positives.csv ({counts['true_positive']} videos)")
    print(f"  - true_negatives.csv ({counts['true_negative']} videos)")
    print(f"  - all_correct_predictions.csv ({correct} videos)")


if __name__ == '__main__':
    main()
