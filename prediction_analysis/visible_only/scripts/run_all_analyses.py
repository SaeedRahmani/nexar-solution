"""
Master script to run all prediction analyses - VISIBLE ONLY SUBSET

This script runs all analyses for the subset that excludes videos where
"Target not visible" is True in ANY of the annotation rounds (R1, R2, or R3).

Supports both VAL and ALL datasets.

This script runs:
1. create_filtered_labels.py - Create filtered labels (visible only)
2. extract_predictions.py - Extract and categorize predictions
3. analyze_wrong_predictions.py - Analyze false negatives and false positives
4. analyze_correct_predictions.py - Analyze true positives and true negatives
5. comparative_analysis.py - Compare wrong vs correct predictions
6. time_diff_comparison_with_ratio.py - Time difference comparison with ratio line
7. scenario_distribution_analysis.py - Scenario distribution and performance
8. accuracy_by_scenario.py - Accuracy per scenario with confusion matrix

SUBSET: Excludes videos where "Target not visible" is True in ANY round.

Usage:
    python prediction_analysis/visible_only/scripts/run_all_analyses.py [--dataset VAL|ALL|BOTH]

Output structure:
    prediction_analysis/visible_only/
    ├── scripts/                           - Analysis scripts
    ├── filtered_labels/                   - Filtered scenario labels
    ├── VAL/                               - VAL dataset results
    │   ├── wrong_predictions/
    │   ├── correct_predictions/
    │   ├── comparative_analysis/
    │   ├── scenario_distribution/
    │   └── accuracy_per_scenario_VAL/
    └── ALL/                               - ALL dataset results
        ├── wrong_predictions/
        ├── correct_predictions/
        ├── comparative_analysis/
        ├── scenario_distribution/
        └── accuracy_per_scenario_ALL/
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/visible_only/scripts")


def run_script(script_name, args=None):
    """Run a Python script and return success status."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}" + (f" {' '.join(args)}" if args else ""))
    print("=" * 70)
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    return result.returncode == 0


def run_for_dataset(dataset):
    """Run all analyses for a specific dataset (VAL or ALL)."""
    
    print("\n" + "=" * 70)
    print(f"RUNNING ANALYSES FOR {dataset} DATASET")
    print("=" * 70)
    
    # Scripts that need dataset argument
    dataset_scripts = [
        ("extract_predictions.py", "Extract and categorize predictions"),
        ("accuracy_by_scenario.py", "Accuracy per scenario with confusion matrix"),
    ]
    
    results = {}
    
    for script, description in dataset_scripts:
        print(f"\n{'='*70}")
        print(f"Step: {description} - {dataset}")
        print(f"{'='*70}")
        
        success = run_script(script, ['--dataset', dataset])
        results[f"{script} ({dataset})"] = success
        
        if not success:
            print(f"\n⚠️  Warning: {script} may have encountered issues.")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BOTH', choices=['VAL', 'ALL', 'BOTH'],
                        help='Dataset to analyze: VAL, ALL, or BOTH')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREDICTION ANALYSIS - VISIBLE ONLY SUBSET")
    print("(Excluding videos where 'Target not visible' is True in ANY round)")
    print("=" * 70)
    print("\nThis will run all prediction analyses for the visible only subset.")
    
    all_results = {}
    
    # Step 1: Create filtered labels (only once)
    print(f"\n{'='*70}")
    print("Step 1: Create filtered labels (visible only)")
    print(f"{'='*70}")
    success = run_script("create_filtered_labels.py")
    all_results["create_filtered_labels.py"] = success
    
    # Step 2: Run analyses for each dataset
    datasets = ['VAL', 'ALL'] if args.dataset == 'BOTH' else [args.dataset]
    
    for dataset in datasets:
        results = run_for_dataset(dataset)
        all_results.update(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    
    for script, success in all_results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {script}")
    
    print("\n" + "-" * 70)
    print("OUTPUT STRUCTURE:")
    print("-" * 70)
    print("""
prediction_analysis/visible_only/
├── scripts/
│   ├── run_all_analyses.py           (this script)
│   ├── create_filtered_labels.py
│   ├── extract_predictions.py    
│   └── accuracy_by_scenario.py
├── filtered_labels/
│   ├── final_matched_visible_only.csv
│   └── excluded_not_visible_videos.csv
├── VAL/
│   ├── wrong_predictions/
│   │   ├── false_negatives.csv
│   │   ├── false_positives.csv
│   │   └── all_wrong_predictions.csv
│   ├── correct_predictions/
│   │   ├── true_positives.csv
│   │   ├── true_negatives.csv
│   │   └── all_correct_predictions.csv
│   └── accuracy_per_scenario_VAL/
│       ├── accuracy_level1.png/pdf
│       ├── accuracy_level2.png/pdf
│       ├── confusion_matrix.png/pdf
│       └── accuracy_report.txt
└── ALL/
    ├── wrong_predictions/
    ├── correct_predictions/
    └── accuracy_per_scenario_ALL/
        ├── accuracy_level1.png/pdf
        ├── accuracy_level2.png/pdf
        ├── confusion_matrix.png/pdf
        └── accuracy_report.txt
""")


if __name__ == '__main__':
    main()
