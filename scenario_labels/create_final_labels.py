import pandas as pd
import os
import sys

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compare_rounds():
    """
    Compare labeling rounds and create comprehensive analysis:
    1. Find mismatches between round 1 and round 2 (Level 1, Level 2, Any)
    2. Create CSV for videos matching at both levels
    3. For mismatches, check if round 3 sanity check has labels
    4. Create final matched CSV with round 3 labels where applicable
    5. Create CSV for unresolved mismatches (no round 3 label)
    """
    round1_path = os.path.join(PROJECT_ROOT, 'scenario_labels/round_1.csv')
    round2_path = os.path.join(PROJECT_ROOT, 'scenario_labels/round_2.csv')
    round3_path = os.path.join(PROJECT_ROOT, 'scenario_labels/round_3_sanityCheck.csv')
    output_dir = os.path.join(PROJECT_ROOT, 'scenario_labels')

    # Check if files exist
    if not os.path.exists(round1_path):
        print(f"Error: Round 1 file not found: {round1_path}")
        return
    if not os.path.exists(round2_path):
        print(f"Error: Round 2 file not found: {round2_path}")
        return
    if not os.path.exists(round3_path):
        print(f"Error: Round 3 file not found: {round3_path}")
        return

    # Load data
    df1 = pd.read_csv(round1_path)
    df2 = pd.read_csv(round2_path)
    df3 = pd.read_csv(round3_path)

    # Clean column names (strip whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()

    # Ensure Video Name is string and strip whitespace
    df1['Video Name'] = df1['Video Name'].astype(str).str.strip()
    df2['Video Name'] = df2['Video Name'].astype(str).str.strip()
    df3['Video Name'] = df3['Video Name'].astype(str).str.strip()

    # Select relevant columns
    cols = ['Video Name', 'Scenario Level 1', 'Scenario Level 2']

    # Check if columns exist
    for col in cols:
        if col not in df1.columns:
            print(f"Error: Column '{col}' not found in Round 1 file.")
            return
        if col not in df2.columns:
            print(f"Error: Column '{col}' not found in Round 2 file.")
            return
        if col not in df3.columns:
            print(f"Error: Column '{col}' not found in Round 3 file.")
            return

    # Prepare dataframes with renamed columns
    df1_sel = df1[cols].rename(columns={'Scenario Level 1': 'L1_R1', 'Scenario Level 2': 'L2_R1'})
    df2_sel = df2[cols].rename(columns={'Scenario Level 1': 'L1_R2', 'Scenario Level 2': 'L2_R2'})
    df3_sel = df3[cols].rename(columns={'Scenario Level 1': 'L1_R3', 'Scenario Level 2': 'L2_R3'})

    # Fill NaNs with empty string to allow comparison
    df1_sel = df1_sel.fillna('')
    df2_sel = df2_sel.fillna('')
    df3_sel = df3_sel.fillna('')

    # Strip whitespace from values
    for col in ['L1_R1', 'L2_R1']:
        df1_sel[col] = df1_sel[col].astype(str).str.strip()
    for col in ['L1_R2', 'L2_R2']:
        df2_sel[col] = df2_sel[col].astype(str).str.strip()
    for col in ['L1_R3', 'L2_R3']:
        df3_sel[col] = df3_sel[col].astype(str).str.strip()

    # Remove duplicates based on Video Name (keep first occurrence)
    df1_sel = df1_sel.drop_duplicates(subset=['Video Name'], keep='first')
    df2_sel = df2_sel.drop_duplicates(subset=['Video Name'], keep='first')
    df3_sel = df3_sel.drop_duplicates(subset=['Video Name'], keep='first')

    # Calculate unique videos
    videos1 = set(df1_sel['Video Name'])
    videos2 = set(df2_sel['Video Name'])
    videos3 = set(df3_sel['Video Name'])
    common_videos_r1_r2 = videos1.intersection(videos2)
    all_unique_videos = videos1.union(videos2)

    print("=" * 60)
    print("ROUND COMPARISON ANALYSIS")
    print("=" * 60)
    print(f"\nTotal rows in Round 1: {len(df1)}")
    print(f"Total rows in Round 2: {len(df2)}")
    print(f"Total rows in Round 3 (Sanity Check): {len(df3)}")
    print(f"\nUnique videos in Round 1: {len(videos1)}")
    print(f"Unique videos in Round 2: {len(videos2)}")
    print(f"Unique videos in Round 3: {len(videos3)}")
    print(f"\nUnique videos common to Round 1 & 2: {len(common_videos_r1_r2)}")
    print(f"Total unique videos across Round 1 & 2 (Union): {len(all_unique_videos)}")
    print("-" * 60)

    # Merge Round 1 and Round 2 on Video Name
    merged = pd.merge(df1_sel, df2_sel, on='Video Name', how='inner')

    # Remove duplicates if any
    merged = merged.drop_duplicates(subset=['Video Name'])

    print(f"\nCommon videos found (after merge): {len(merged)}")
    print("-" * 60)

    # Level 1 Comparison
    merged['L1_Match'] = merged['L1_R1'] == merged['L1_R2']
    l1_matches = merged[merged['L1_Match']]
    l1_mismatches = merged[~merged['L1_Match']]

    print(f"\n--- LEVEL 1 COMPARISON ---")
    print(f"Level 1 Matches: {len(l1_matches)}")
    print(f"Level 1 Mismatches: {len(l1_mismatches)}")

    if len(l1_mismatches) > 0:
        print("\nLevel 1 Mismatches (First 10):")
        print(l1_mismatches[['Video Name', 'L1_R1', 'L1_R2']].head(10).to_string(index=False))

        # Save full mismatch report
        l1_mismatches[['Video Name', 'L1_R1', 'L1_R2']].to_csv(
            f'{output_dir}/level_1_mismatches.csv', index=False)
        print(f"\nFull Level 1 mismatch report saved to '{output_dir}/level_1_mismatches.csv'")

    # Level 2 Comparison
    merged['L2_Match'] = merged['L2_R1'] == merged['L2_R2']
    l2_matches = merged[merged['L2_Match']]
    l2_mismatches = merged[~merged['L2_Match']]

    print(f"\n--- LEVEL 2 COMPARISON ---")
    print(f"Level 2 Matches: {len(l2_matches)}")
    print(f"Level 2 Mismatches: {len(l2_mismatches)}")

    if len(l2_mismatches) > 0:
        print("\nLevel 2 Mismatches (First 10):")
        print(l2_mismatches[['Video Name', 'L2_R1', 'L2_R2']].head(10).to_string(index=False))

        # Save full mismatch report
        l2_mismatches[['Video Name', 'L2_R1', 'L2_R2']].to_csv(
            f'{output_dir}/level_2_mismatches.csv', index=False)
        print(f"\nFull Level 2 mismatch report saved to '{output_dir}/level_2_mismatches.csv'")

    # Any Mismatch (Union of L1 and L2 mismatches)
    merged['Any_Mismatch'] = (~merged['L1_Match']) | (~merged['L2_Match'])
    any_mismatches = merged[merged['Any_Mismatch']]
    both_match = merged[~merged['Any_Mismatch']]

    print(f"\n--- ANY MISMATCH (L1 OR L2) ---")
    print(f"Total videos with ANY Mismatch: {len(any_mismatches)}")
    print(f"Total videos with BOTH L1 and L2 Match: {len(both_match)}")

    if len(any_mismatches) > 0:
        # Save full any mismatch report
        any_mismatches[['Video Name', 'L1_R1', 'L1_R2', 'L1_Match', 'L2_R1', 'L2_R2', 'L2_Match']].to_csv(
            f'{output_dir}/any_mismatches.csv', index=False)
        print(f"Full Any Mismatch report saved to '{output_dir}/any_mismatches.csv'")

    # Save full comparison report
    report_df = merged[['Video Name', 'L1_R1', 'L1_R2', 'L1_Match', 'L2_R1', 'L2_R2', 'L2_Match']]
    report_df.to_csv(f'{output_dir}/comparison_report.csv', index=False)
    print(f"\nFull comparison report saved to '{output_dir}/comparison_report.csv'")

    print("\n" + "=" * 60)
    print("MATCHED VIDEOS ANALYSIS (Both L1 and L2 Match)")
    print("=" * 60)

    # Create CSV for videos that match at both levels
    # Use the labels from Round 2 (or Round 1, they're the same since they match)
    if len(both_match) > 0:
        matched_output = both_match[['Video Name', 'L1_R2', 'L2_R2']].copy()
        matched_output = matched_output.rename(columns={
            'L1_R2': 'Scenario Level 1',
            'L2_R2': 'Scenario Level 2'
        })
        matched_output['Source'] = 'R1_R2_Match'
        matched_output.to_csv(f'{output_dir}/matched_both_levels.csv', index=False)
        print(f"\nVideos matching at both levels: {len(matched_output)}")
        print(f"Saved to '{output_dir}/matched_both_levels.csv'")

    print("\n" + "=" * 60)
    print("ROUND 3 SANITY CHECK ANALYSIS")
    print("=" * 60)

    # Get the mismatched video names
    mismatched_video_names = set(any_mismatches['Video Name'])
    print(f"\nTotal mismatched videos from R1 vs R2: {len(mismatched_video_names)}")

    # Check which mismatched videos have labels in round 3
    mismatched_with_r3 = df3_sel[df3_sel['Video Name'].isin(mismatched_video_names)]
    mismatched_video_names_with_r3 = set(mismatched_with_r3['Video Name'])
    mismatched_video_names_without_r3 = mismatched_video_names - mismatched_video_names_with_r3

    print(f"Mismatched videos WITH Round 3 label: {len(mismatched_video_names_with_r3)}")
    print(f"Mismatched videos WITHOUT Round 3 label: {len(mismatched_video_names_without_r3)}")

    # Create resolved mismatches output (using Round 3 labels)
    resolved_from_r3 = pd.DataFrame()
    if len(mismatched_with_r3) > 0:
        resolved_from_r3 = mismatched_with_r3[['Video Name', 'L1_R3', 'L2_R3']].copy()
        resolved_from_r3 = resolved_from_r3.rename(columns={
            'L1_R3': 'Scenario Level 1',
            'L2_R3': 'Scenario Level 2'
        })
        resolved_from_r3['Source'] = 'R3_SanityCheck'
        resolved_from_r3.to_csv(f'{output_dir}/mismatches_resolved_by_r3.csv', index=False)
        print(f"\nMismatches resolved by Round 3 saved to '{output_dir}/mismatches_resolved_by_r3.csv'")

    # Create unresolved mismatches output (no Round 3 label)
    if len(mismatched_video_names_without_r3) > 0:
        unresolved = any_mismatches[any_mismatches['Video Name'].isin(mismatched_video_names_without_r3)]
        unresolved[['Video Name', 'L1_R1', 'L1_R2', 'L2_R1', 'L2_R2']].to_csv(
            f'{output_dir}/unresolved_mismatches.csv', index=False)
        print(f"Unresolved mismatches saved to '{output_dir}/unresolved_mismatches.csv'")
        print(f"\nUnresolved mismatches (First 10):")
        print(unresolved[['Video Name', 'L1_R1', 'L1_R2', 'L2_R1', 'L2_R2']].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("VIDEOS ONLY IN ONE ROUND")
    print("=" * 60)

    # Find videos only in R1 or only in R2
    only_in_r1 = videos1 - videos2
    only_in_r2 = videos2 - videos1
    only_one_round = only_in_r1 | only_in_r2

    print(f"\nVideos only in R1: {len(only_in_r1)}")
    if only_in_r1:
        print(f"  {sorted(only_in_r1)}")
    print(f"Videos only in R2: {len(only_in_r2)}")
    if only_in_r2:
        print(f"  {sorted(only_in_r2)}")

    # Check if these have R3 labels
    only_one_round_with_r3 = df3_sel[df3_sel['Video Name'].isin(only_one_round)]
    only_one_round_names_with_r3 = set(only_one_round_with_r3['Video Name'])
    only_one_round_names_without_r3 = only_one_round - only_one_round_names_with_r3

    print(f"\nVideos with only one round label BUT have R3 label: {len(only_one_round_names_with_r3)}")
    print(f"Videos with only one round label and NO R3 label: {len(only_one_round_names_without_r3)}")

    # Create output for videos only in one round resolved by R3
    resolved_one_round_from_r3 = pd.DataFrame()
    if len(only_one_round_with_r3) > 0:
        resolved_one_round_from_r3 = only_one_round_with_r3[['Video Name', 'L1_R3', 'L2_R3']].copy()
        resolved_one_round_from_r3 = resolved_one_round_from_r3.rename(columns={
            'L1_R3': 'Scenario Level 1',
            'L2_R3': 'Scenario Level 2'
        })
        resolved_one_round_from_r3['Source'] = 'R3_OneRound'

    print("\n" + "=" * 60)
    print("FINAL COMBINED MATCHED CSV")
    print("=" * 60)

    # Create combined matched CSV:
    # 1. Videos that matched at both levels in R1 vs R2
    # 2. Videos that mismatched in R1 vs R2 but have Round 3 label
    # 3. Videos only in one round but have Round 3 label
    final_matched_list = []

    # Add R1/R2 matched videos
    if len(both_match) > 0:
        matched_r1r2 = both_match[['Video Name', 'L1_R2', 'L2_R2']].copy()
        matched_r1r2 = matched_r1r2.rename(columns={
            'L1_R2': 'Scenario Level 1',
            'L2_R2': 'Scenario Level 2'
        })
        matched_r1r2['Source'] = 'R1_R2_Match'
        final_matched_list.append(matched_r1r2)

    # Add R3 resolved mismatches
    if len(resolved_from_r3) > 0:
        final_matched_list.append(resolved_from_r3)

    # Add R3 resolved one-round videos
    if len(resolved_one_round_from_r3) > 0:
        final_matched_list.append(resolved_one_round_from_r3)

    if final_matched_list:
        final_matched = pd.concat(final_matched_list, ignore_index=True)
        final_matched = final_matched.drop_duplicates(subset=['Video Name'], keep='first')
        final_matched.to_csv(f'{output_dir}/final_matched_all.csv', index=False)
        
        r1r2_count = len(final_matched[final_matched['Source'] == 'R1_R2_Match'])
        r3_mismatch_count = len(final_matched[final_matched['Source'] == 'R3_SanityCheck'])
        r3_oneround_count = len(final_matched[final_matched['Source'] == 'R3_OneRound'])
        
        print(f"\nFinal combined matched videos: {len(final_matched)}")
        print(f"  - From R1/R2 match: {r1r2_count}")
        print(f"  - From R3 (resolved mismatches): {r3_mismatch_count}")
        print(f"  - From R3 (one-round videos): {r3_oneround_count}")
        print(f"Saved to '{output_dir}/final_matched_all.csv'")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal videos in analysis (R1 ∩ R2): {len(merged)}")
    print(f"Videos matched at both levels (R1 = R2): {len(both_match)}")
    print(f"Videos with any mismatch (R1 ≠ R2): {len(any_mismatches)}")
    print(f"  - Resolved by Round 3: {len(mismatched_video_names_with_r3)}")
    print(f"  - Unresolved (need review): {len(mismatched_video_names_without_r3)}")
    print(f"\nVideos only in one round: {len(only_one_round)}")
    print(f"  - Resolved by Round 3: {len(only_one_round_names_with_r3)}")
    print(f"  - Unresolved (need review): {len(only_one_round_names_without_r3)}")
    print(f"\nTotal resolved videos: {len(both_match) + len(mismatched_video_names_with_r3) + len(only_one_round_names_with_r3)}")
    print(f"Total unresolved videos: {len(mismatched_video_names_without_r3) + len(only_one_round_names_without_r3)}")

    print("\n" + "=" * 60)
    print("OUTPUT FILES GENERATED:")
    print("=" * 60)
    print(f"  1. {output_dir}/level_1_mismatches.csv - Level 1 mismatches between R1 and R2")
    print(f"  2. {output_dir}/level_2_mismatches.csv - Level 2 mismatches between R1 and R2")
    print(f"  3. {output_dir}/any_mismatches.csv - Any mismatch (L1 or L2) between R1 and R2")
    print(f"  4. {output_dir}/comparison_report.csv - Full comparison report")
    print(f"  5. {output_dir}/matched_both_levels.csv - Videos matching at both levels")
    print(f"  6. {output_dir}/mismatches_resolved_by_r3.csv - Mismatches with R3 labels")
    print(f"  7. {output_dir}/unresolved_mismatches.csv - Mismatches without R3 labels")
    print(f"  8. {output_dir}/final_matched_all.csv - Combined: R1/R2 matches + R3 resolved")


if __name__ == "__main__":
    compare_rounds()
