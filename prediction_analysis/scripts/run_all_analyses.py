"""
Master script to run all prediction analyses.

This script runs:
1. extract_predictions.py - Extract and categorize predictions
2. analyze_wrong_predictions.py - Analyze false negatives and false positives
3. analyze_correct_predictions.py - Analyze true positives and true negatives
4. comparative_analysis.py - Compare wrong vs correct predictions

Usage:
    python prediction_analysis/scripts/run_all_analyses.py

Output structure:
    prediction_analysis/
    ├── scripts/               - Analysis scripts
    ├── wrong_predictions/     - Wrong prediction data and visualizations
    ├── correct_predictions/   - Correct prediction data and visualizations
    └── comparative_analysis/  - Comparative visualizations and statistics
"""

import os
import sys
import subprocess

PROJECT_ROOT = "/home/sra2157/git/nexar-solution"
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "prediction_analysis/scripts")


def run_script(script_name):
    """Run a Python script and return success status."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    print("=" * 70)
    print("PREDICTION ANALYSIS - MASTER SCRIPT")
    print("=" * 70)
    print("\nThis will run all prediction analyses in sequence.")
    print("Data source: video_predictions_ALL/")
    
    scripts = [
        ("extract_predictions.py", "Extract and categorize predictions"),
        ("analyze_wrong_predictions.py", "Analyze wrong predictions (FN, FP)"),
        ("analyze_correct_predictions.py", "Analyze correct predictions (TP, TN)"),
        ("comparative_analysis.py", "Compare wrong vs correct predictions"),
    ]
    
    results = {}
    
    for script, description in scripts:
        print(f"\n{'='*70}")
        print(f"Step: {description}")
        print(f"{'='*70}")
        
        success = run_script(script)
        results[script] = success
        
        if not success:
            print(f"\n⚠️  Warning: {script} may have encountered issues.")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    
    for script, success in results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {script}")
    
    print("\n" + "-" * 70)
    print("OUTPUT STRUCTURE:")
    print("-" * 70)
    print("""
prediction_analysis/
├── scripts/
│   ├── run_all_analyses.py      (this script)
│   ├── extract_predictions.py    
│   ├── analyze_wrong_predictions.py
│   ├── analyze_correct_predictions.py
│   └── comparative_analysis.py
├── wrong_predictions/
│   ├── false_negatives.csv       (⚠️ CRITICAL - missed accidents)
│   ├── false_positives.csv       (false alarms)
│   ├── all_wrong_predictions.csv
│   ├── time_difference_analysis.png/pdf
│   ├── prediction_confidence_analysis.png/pdf
│   └── summary_statistics.txt
├── correct_predictions/
│   ├── true_positives.csv
│   ├── true_negatives.csv
│   ├── all_correct_predictions.csv
│   ├── time_difference_analysis.png/pdf
│   ├── prediction_confidence_analysis.png/pdf
│   └── summary_statistics.txt
└── comparative_analysis/
    ├── time_diff_comparison_histograms.png/pdf
    ├── time_diff_comparison_boxplots.png/pdf
    ├── time_diff_comparison_violin.png/pdf
    ├── confidence_comparison.png/pdf
    └── statistical_comparison.txt
""")


if __name__ == '__main__':
    main()
