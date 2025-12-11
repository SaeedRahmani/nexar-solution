"""
Script to extract time differences between time_of_event and time_of_alert
for videos that were wrongly predicted by the non-focal model.

Only false negatives (positive videos predicted as negative) will have timing data,
since negative videos don't have time_of_event and time_of_alert values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
train_csv_path = "/home/sra2157/git/nexar-solution/dataset/train.csv"
wrong_train_path = "/home/sra2157/git/nexar-solution/wrong_predictions/wrong_predictions_train.csv"
wrong_val_path = "/home/sra2157/git/nexar-solution/wrong_predictions/wrong_predictions_val.csv"
output_dir = "/home/sra2157/git/nexar-solution/wrong_predictions"

# Load the training data with timing information
train_df = pd.read_csv(train_csv_path)
train_df['id'] = train_df['id'].apply(lambda x: f"{x:05d}")  # Format as 5-digit string

# Load wrong predictions (ensure video_id is string type)
wrong_train = pd.read_csv(wrong_train_path, dtype={'video_id': str})
wrong_val = pd.read_csv(wrong_val_path, dtype={'video_id': str})

# Combine all wrong predictions
all_wrong = pd.concat([wrong_train, wrong_val], ignore_index=True)

# Merge with train data to get timing information
merged = all_wrong.merge(
    train_df[['id', 'time_of_event', 'time_of_alert', 'target']],
    left_on='video_id',
    right_on='id',
    how='left'
)

# Calculate time difference (time_of_event - time_of_alert)
# This represents how much earlier the alert was compared to the event
merged['time_difference'] = merged['time_of_event'] - merged['time_of_alert']

# Select and rename columns for output
output_df = merged[[
    'video_id', 'pred', 'prob', 'true_label', 'error_type',
    'time_of_event', 'time_of_alert', 'time_difference'
]].copy()

# Sort by error type and video_id
output_df = output_df.sort_values(['error_type', 'video_id'])

# Save complete output
output_df.to_csv(f"{output_dir}/wrong_predictions_with_timing.csv", index=False)

# Create separate outputs for false negatives only (these are the ones with timing data)
false_negatives = output_df[output_df['error_type'] == 'false_negative'].dropna(subset=['time_of_event'])
false_negatives.to_csv(f"{output_dir}/false_negatives_timing.csv", index=False)

# Create separate output for false positives (no timing data - negative videos)
false_positives = output_df[output_df['error_type'] == 'false_positive'].copy()
false_positives_output = false_positives[['video_id', 'pred', 'prob', 'true_label', 'error_type']]
false_positives_output.to_csv(f"{output_dir}/false_positives_no_timing.csv", index=False)

# Print summary
print("=" * 60)
print("Time Difference Analysis for Wrong Predictions")
print("=" * 60)

print(f"\nTotal wrong predictions: {len(output_df)}")
print(f"  - False negatives (with timing data): {len(false_negatives)}")
print(f"  - False positives (no timing data - negative videos): {len(output_df[output_df['error_type'] == 'false_positive'])}")

print("\n" + "=" * 60)
print("FALSE NEGATIVES (Positive videos incorrectly predicted as negative)")
print("These have time_of_event and time_of_alert data")
print("=" * 60)

if len(false_negatives) > 0:
    print(f"\nStatistics for time difference (event - alert):")
    print(f"  Mean: {false_negatives['time_difference'].mean():.3f} seconds")
    print(f"  Std:  {false_negatives['time_difference'].std():.3f} seconds")
    print(f"  Min:  {false_negatives['time_difference'].min():.3f} seconds")
    print(f"  Max:  {false_negatives['time_difference'].max():.3f} seconds")
    
    print("\n\nDetailed breakdown of false negatives:")
    print("-" * 80)
    print(f"{'Video ID':<10} {'Event Time':<12} {'Alert Time':<12} {'Diff (sec)':<12} {'Pred Prob':<10}")
    print("-" * 80)
    
    for _, row in false_negatives.iterrows():
        print(f"{row['video_id']:<10} {row['time_of_event']:<12.3f} {row['time_of_alert']:<12.3f} {row['time_difference']:<12.3f} {row['prob']:<10.4f}")

print("\n" + "=" * 60)
print("FALSE POSITIVES (Negative videos incorrectly predicted as positive)")
print("These do NOT have time_of_event and time_of_alert data")
print("=" * 60)
print(f"\nTotal false positives: {len(false_positives)}")
print(f"\nFirst 10 false positives:")
print("-" * 50)
print(f"{'Video ID':<10} {'Pred Prob':<12}")
print("-" * 50)
for _, row in false_positives.head(10).iterrows():
    print(f"{row['video_id']:<10} {row['prob']:<12.4f}")

print("\n" + "=" * 60)
print("Files saved:")
print(f"  1. {output_dir}/wrong_predictions_with_timing.csv")
print(f"     (All wrong predictions with timing info where available)")
print(f"  2. {output_dir}/false_negatives_timing.csv")
print(f"     (Only false negatives with complete timing data - {len(false_negatives)} videos)")
print(f"  3. {output_dir}/false_positives_no_timing.csv")
print(f"     (False positives without timing data - {len(false_positives)} videos)")
print("=" * 60)

# ============================================================
# PLOTTING
# ============================================================

if len(false_negatives) > 0:
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'light': '#C73E1D',
        'dark': '#3B1F2B'
    }
    
    # ---- Plot 1: Histogram of Time Differences (Top Left) ----
    ax1 = fig.add_subplot(2, 2, 1)
    n, bins, patches = ax1.hist(false_negatives['time_difference'], bins=10, 
                                 color=colors['primary'], edgecolor='white', 
                                 linewidth=1.2, alpha=0.8)
    ax1.axvline(false_negatives['time_difference'].mean(), color=colors['light'], 
                linestyle='--', linewidth=2, label=f'Mean: {false_negatives["time_difference"].mean():.2f}s')
    ax1.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Time Differences\n(Event Time - Alert Time)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ---- Plot 2: Box Plot + Strip Plot (Top Right) ----
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Box plot
    bp = ax2.boxplot(false_negatives['time_difference'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor=colors['primary'], alpha=0.6),
                     medianprops=dict(color=colors['light'], linewidth=2),
                     whiskerprops=dict(color=colors['dark'], linewidth=1.5),
                     capprops=dict(color=colors['dark'], linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor=colors['accent'], markersize=8))
    
    # Add individual points (strip plot style)
    x_jitter = np.random.normal(1, 0.04, size=len(false_negatives))
    ax2.scatter(x_jitter, false_negatives['time_difference'], 
                c=colors['accent'], s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Difference Distribution\n(Box Plot with Individual Points)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['False Negatives'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add statistics annotation
    stats_text = (f"n = {len(false_negatives)}\n"
                  f"Mean = {false_negatives['time_difference'].mean():.2f}s\n"
                  f"Median = {false_negatives['time_difference'].median():.2f}s\n"
                  f"Std = {false_negatives['time_difference'].std():.2f}s")
    ax2.annotate(stats_text, xy=(1.3, false_negatives['time_difference'].max()), 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ---- Plot 3: Bar Chart by Video ID (Bottom Left) ----
    ax3 = fig.add_subplot(2, 2, 3)
    sorted_fn = false_negatives.sort_values('time_difference', ascending=True)
    bars = ax3.barh(sorted_fn['video_id'], sorted_fn['time_difference'], 
                    color=colors['secondary'], edgecolor='white', linewidth=0.5, alpha=0.85)
    ax3.axvline(false_negatives['time_difference'].mean(), color=colors['light'], 
                linestyle='--', linewidth=2, label=f'Mean: {false_negatives["time_difference"].mean():.2f}s')
    ax3.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Video ID', fontsize=12, fontweight='bold')
    ax3.set_title('Time Difference by Video\n(False Negatives)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ---- Plot 4: Scatter Plot - Time Difference vs Prediction Probability (Bottom Right) ----
    ax4 = fig.add_subplot(2, 2, 4)
    # Use RdYlGn (not reversed) so that:
    # - Red = low probability = model was confident but WRONG (bad!)
    # - Green = high probability = model was uncertain (less bad)
    scatter = ax4.scatter(false_negatives['time_difference'], false_negatives['prob'], 
                          c=false_negatives['prob'], cmap='RdYlGn', 
                          s=150, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add video ID labels
    for _, row in false_negatives.iterrows():
        ax4.annotate(row['video_id'], (row['time_difference'], row['prob']),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)
    
    ax4.set_xlabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Time Difference vs Model Confidence\n(Red = confident wrong predictions = BAD)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Decision boundary (0.5)')
    ax4.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Prediction Probability\n(Red=confident wrong, Green=uncertain)', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_difference_analysis.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/time_difference_analysis.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\n  3. {output_dir}/time_difference_analysis.png")
    print(f"     (Visualization of time differences)")
    print(f"  4. {output_dir}/time_difference_analysis.pdf")
    print(f"     (PDF version of visualization)")
    
    plt.show()
    print("\nPlots generated successfully!")
