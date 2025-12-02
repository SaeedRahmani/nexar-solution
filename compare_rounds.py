import pandas as pd
import os

def compare_rounds():
    round1_path = 'scenario_labels/round_1.csv'
    round2_path = 'scenario_labels/round_2.csv'

    if not os.path.exists(round1_path) or not os.path.exists(round2_path):
        print("Error: One or both input files not found.")
        return

    df1 = pd.read_csv(round1_path)
    df2 = pd.read_csv(round2_path)

    # Ensure Video Name is string and strip whitespace
    df1['Video Name'] = df1['Video Name'].astype(str).str.strip()
    df2['Video Name'] = df2['Video Name'].astype(str).str.strip()

    # Select relevant columns
    cols = ['Video Name', 'Scenario Level 1', 'Scenario Level 2']
    
    # Clean column names (strip whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Check if columns exist
    for col in cols:
        if col not in df1.columns:
            print(f"Error: Column '{col}' not found in Round 1 file.")
            return
        if col not in df2.columns:
            print(f"Error: Column '{col}' not found in Round 2 file.")
            return

    df1 = df1[cols].rename(columns={'Scenario Level 1': 'L1_R1', 'Scenario Level 2': 'L2_R1'})
    df2 = df2[cols].rename(columns={'Scenario Level 1': 'L1_R2', 'Scenario Level 2': 'L2_R2'})

    # Fill NaNs with empty string to allow comparison
    df1 = df1.fillna('')
    df2 = df2.fillna('')

    # Strip whitespace from values
    for col in ['L1_R1', 'L2_R1']:
        df1[col] = df1[col].astype(str).str.strip()
    for col in ['L1_R2', 'L2_R2']:
        df2[col] = df2[col].astype(str).str.strip()

    # Calculate unique videos intersection
    videos1 = set(df1['Video Name'])
    videos2 = set(df2['Video Name'])
    common_videos_set = videos1.intersection(videos2)
    all_unique_videos = videos1.union(videos2)

    print(f"Total rows in Round 1: {len(df1)}")
    print(f"Total rows in Round 2: {len(df2)}")
    print(f"Unique videos in Round 1: {len(videos1)}")
    print(f"Unique videos in Round 2: {len(videos2)}")
    print(f"Unique videos common to both rounds: {len(common_videos_set)}")
    print(f"Total unique videos across both rounds (Union): {len(all_unique_videos)}")
    print("-" * 30)

    # Merge on Video Name
    merged = pd.merge(df1, df2, on='Video Name', how='inner')

    # Remove duplicates if any (though set intersection above confirms unique count)
    merged = merged.drop_duplicates(subset=['Video Name'])

    print(f"Common videos found (after merge): {len(merged)}")
    print("-" * 30)

    # Level 1 Comparison
    merged['L1_Match'] = merged['L1_R1'] == merged['L1_R2']
    l1_matches = merged[merged['L1_Match']]
    l1_mismatches = merged[~merged['L1_Match']]

    print(f"Level 1 Matches: {len(l1_matches)}")
    print(f"Level 1 Mismatches: {len(l1_mismatches)}")
    
    if len(l1_mismatches) > 0:
        print("\nLevel 1 Mismatches (First 10):")
        print(l1_mismatches[['Video Name', 'L1_R1', 'L1_R2']].head(10).to_string(index=False))
        
        # Save full mismatch report
        l1_mismatches[['Video Name', 'L1_R1', 'L1_R2']].to_csv('scenario_labels/level_1_mismatches.csv', index=False)
        print("\nFull Level 1 mismatch report saved to 'scenario_labels/level_1_mismatches.csv'")

    print("-" * 30)

    # Level 2 Comparison
    merged['L2_Match'] = merged['L2_R1'] == merged['L2_R2']
    l2_matches = merged[merged['L2_Match']]
    l2_mismatches = merged[~merged['L2_Match']]

    print(f"Level 2 Matches: {len(l2_matches)}")
    print(f"Level 2 Mismatches: {len(l2_mismatches)}")

    if len(l2_mismatches) > 0:
        print("\nLevel 2 Mismatches (First 10):")
        print(l2_mismatches[['Video Name', 'L2_R1', 'L2_R2']].head(10).to_string(index=False))
        
        # Save full mismatch report
        l2_mismatches[['Video Name', 'L2_R1', 'L2_R2']].to_csv('scenario_labels/level_2_mismatches.csv', index=False)
        print("\nFull Level 2 mismatch report saved to 'scenario_labels/level_2_mismatches.csv'")

    print("-" * 30)

    # Any Mismatch (Union of L1 and L2 mismatches)
    merged['Any_Mismatch'] = (~merged['L1_Match']) | (~merged['L2_Match'])
    any_mismatches = merged[merged['Any_Mismatch']]

    print(f"Total Unique Videos with ANY Mismatch (L1 or L2): {len(any_mismatches)}")
    
    if len(any_mismatches) > 0:
        # Save full any mismatch report
        any_mismatches[['Video Name', 'L1_R1', 'L1_R2', 'L1_Match', 'L2_R1', 'L2_R2', 'L2_Match']].to_csv('scenario_labels/any_mismatches.csv', index=False)
        print("Full Any Mismatch report saved to 'scenario_labels/any_mismatches.csv'")

    # Create a combined report
    report_df = merged[['Video Name', 'L1_R1', 'L1_R2', 'L1_Match', 'L2_R1', 'L2_R2', 'L2_Match']]
    report_df.to_csv('scenario_labels/comparison_report.csv', index=False)
    print("\nFull comparison report saved to 'scenario_labels/comparison_report.csv'")

if __name__ == "__main__":
    compare_rounds()
