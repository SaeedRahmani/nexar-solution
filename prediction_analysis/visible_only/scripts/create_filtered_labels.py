"""
Create filtered labels excluding "Target not visible" videos.

This script creates a subset of the scenario labels that excludes all videos
where "Target not visible" is True in ANY of the rounds (Round 1, 2, or 3).

Usage:
    python prediction_analysis/visible_only/scripts/create_filtered_labels.py

Output:
    prediction_analysis/visible_only/filtered_labels/final_matched_visible_only.csv
    prediction_analysis/visible_only/filtered_labels/excluded_not_visible_videos.csv
"""

import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
ROUND1_PATH = os.path.join(PROJECT_ROOT, "scenario_labels/round_1.csv")
ROUND2_PATH = os.path.join(PROJECT_ROOT, "scenario_labels/round_2.csv")
ROUND3_PATH = os.path.join(PROJECT_ROOT, "scenario_labels/round_3_sanityCheck.csv")
FINAL_LABELS_PATH = os.path.join(PROJECT_ROOT, "scenario_labels/final_matched_all.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/visible_only/filtered_labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("CREATING FILTERED LABELS")
    print("(Excluding 'Target not visible' from ANY round: R1, R2, or R3)")
    print("=" * 70)
    
    # Load all round files
    print("\nLoading round files...")
    round1_df = pd.read_csv(ROUND1_PATH)
    round2_df = pd.read_csv(ROUND2_PATH)
    round3_df = pd.read_csv(ROUND3_PATH)
    
    print(f"  Round 1: {len(round1_df)} videos, {(round1_df['Target not visible'] == True).sum()} not visible")
    print(f"  Round 2: {len(round2_df)} videos, {(round2_df['Target not visible'] == True).sum()} not visible")
    print(f"  Round 3: {len(round3_df)} videos, {(round3_df['Target not visible'] == True).sum()} not visible")
    
    # Get videos where Target NOT visible in ANY round
    r1_not_visible = set(round1_df[round1_df['Target not visible'] == True]['Video Name'])
    r2_not_visible = set(round2_df[round2_df['Target not visible'] == True]['Video Name'])
    r3_not_visible = set(round3_df[round3_df['Target not visible'] == True]['Video Name'])
    
    all_not_visible = r1_not_visible | r2_not_visible | r3_not_visible
    print(f"\n  Videos with 'Target not visible' in ANY round: {len(all_not_visible)}")
    print(f"    - Only in R1: {len(r1_not_visible - r2_not_visible - r3_not_visible)}")
    print(f"    - Only in R2: {len(r2_not_visible - r1_not_visible - r3_not_visible)}")
    print(f"    - Only in R3: {len(r3_not_visible - r1_not_visible - r2_not_visible)}")
    
    # Get videos where Target IS visible (not in any not_visible set)
    all_videos_r1 = set(round1_df['Video Name'])
    visible_videos = all_videos_r1 - all_not_visible
    
    print(f"\n  Videos with Target VISIBLE (in all rounds): {len(visible_videos)}")
    
    # Load final_matched_all.csv
    print("\nLoading final_matched_all.csv...")
    final_df = pd.read_csv(FINAL_LABELS_PATH)
    print(f"  Total videos in final_matched_all: {len(final_df)}")
    
    # Filter to only include visible videos
    filtered_df = final_df[final_df['Video Name'].isin(visible_videos)]
    print(f"  Videos after filtering: {len(filtered_df)}")
    
    # Save filtered labels
    output_path = os.path.join(OUTPUT_DIR, "final_matched_visible_only.csv")
    filtered_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved filtered labels to: {output_path}")
    
    # Create detailed excluded videos list with source info
    excluded_records = []
    for video in all_not_visible:
        sources = []
        if video in r1_not_visible:
            sources.append('R1')
        if video in r2_not_visible:
            sources.append('R2')
        if video in r3_not_visible:
            sources.append('R3')
        
        # Get scenario info from round1 if available
        r1_info = round1_df[round1_df['Video Name'] == video]
        if len(r1_info) > 0:
            scenario_l1 = r1_info.iloc[0]['Scenario Level 1']
            scenario_l2 = r1_info.iloc[0]['Scenario Level 2']
        else:
            scenario_l1 = 'Unknown'
            scenario_l2 = 'Unknown'
        
        excluded_records.append({
            'Video Name': video,
            'Scenario Level 1': scenario_l1,
            'Scenario Level 2': scenario_l2,
            'Not_Visible_In': ','.join(sources)
        })
    
    excluded_df = pd.DataFrame(excluded_records)
    excluded_path = os.path.join(OUTPUT_DIR, "excluded_not_visible_videos.csv")
    excluded_df.to_csv(excluded_path, index=False)
    print(f"✅ Saved excluded videos list to: {excluded_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original videos: {len(final_df)}")
    print(f"Filtered videos (Target visible in ALL rounds): {len(filtered_df)}")
    print(f"Excluded videos (Target not visible in ANY round): {len(final_df) - len(filtered_df)}")
    
    # Show which scenarios are most affected
    print("\nScenarios most affected by exclusion (Level 1):")
    excluded_scenarios = excluded_df['Scenario Level 1'].value_counts()
    for scenario, count in excluded_scenarios.items():
        print(f"  {scenario}: {count} videos excluded")


if __name__ == "__main__":
    main()
