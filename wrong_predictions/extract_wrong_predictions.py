"""
Script to extract wrong predictions from the non-focal model results.
Creates separate CSV files for training and validation set errors.
"""

import pandas as pd
import os
import glob

# Load the aggregated predictions from non-focal model (ALL dataset)
agg_file = "/home/sra2157/git/nexar-solution/aggregated_results_ALL/Archive_first_run_results/aggregated_predictions_any.csv"
df = pd.read_csv(agg_file)

# Find wrong predictions (where pred != true_label)
wrong_preds = df[df['pred'] != df['true_label']].copy()

# Add error type column
wrong_preds['error_type'] = wrong_preds.apply(
    lambda row: 'false_negative' if row['true_label'] == 1 and row['pred'] == 0 else 'false_positive',
    axis=1
)

# Format video_id as 5-digit string
wrong_preds['video_id_str'] = wrong_preds['video_id'].apply(lambda x: f"{x:05d}")

# Get video IDs from train and val directories by scanning the folder structure
base_dir = "/home/sra2157/git/nexar-solution/balanced_dataset_2s"

# Get train video IDs from both positive and negative folders
train_videos = set()
for label_dir in ['positive', 'negative']:
    train_path = os.path.join(base_dir, 'train', label_dir)
    if os.path.exists(train_path):
        for filename in os.listdir(train_path):
            if filename.endswith('.avi'):
                # Extract video ID from filename like "00128_win120_l1.avi"
                video_id = int(filename.split('_')[0])
                train_videos.add(video_id)

# Get val video IDs from both positive and negative folders
val_videos = set()
for label_dir in ['positive', 'negative']:
    val_path = os.path.join(base_dir, 'val', label_dir)
    if os.path.exists(val_path):
        for filename in os.listdir(val_path):
            if filename.endswith('.avi'):
                video_id = int(filename.split('_')[0])
                val_videos.add(video_id)

# Separate wrong predictions into train and val
wrong_train = wrong_preds[wrong_preds['video_id'].isin(train_videos)].copy()
wrong_val = wrong_preds[wrong_preds['video_id'].isin(val_videos)].copy()

# Create output directory if it doesn't exist
output_dir = "/home/sra2157/git/nexar-solution/wrong_predictions"
os.makedirs(output_dir, exist_ok=True)

# Save training wrong predictions
train_output = wrong_train[['video_id_str', 'pred', 'prob', 'true_label', 'error_type']].rename(
    columns={'video_id_str': 'video_id'}
)
train_output.to_csv(f"{output_dir}/wrong_predictions_train.csv", index=False)

# Save validation wrong predictions
val_output = wrong_val[['video_id_str', 'pred', 'prob', 'true_label', 'error_type']].rename(
    columns={'video_id_str': 'video_id'}
)
val_output.to_csv(f"{output_dir}/wrong_predictions_val.csv", index=False)

# Print summary
print(f"Total wrong predictions: {len(wrong_preds)}")
print(f"\nTraining set wrong predictions: {len(wrong_train)}")
print(f"  - False negatives (missed positives): {len(wrong_train[wrong_train['error_type'] == 'false_negative'])}")
print(f"  - False positives (false alarms): {len(wrong_train[wrong_train['error_type'] == 'false_positive'])}")

print(f"\nValidation set wrong predictions: {len(wrong_val)}")
print(f"  - False negatives (missed positives): {len(wrong_val[wrong_val['error_type'] == 'false_negative'])}")
print(f"  - False positives (false alarms): {len(wrong_val[wrong_val['error_type'] == 'false_positive'])}")

print(f"\nFiles saved:")
print(f"  - {output_dir}/wrong_predictions_train.csv")
print(f"  - {output_dir}/wrong_predictions_val.csv")
