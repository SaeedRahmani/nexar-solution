"""
Master script to run all prediction analyses.

This script runs:
1. extract_predictions.py - Extract and categorize predictions
2. analyze_wrong_predictions.py - Analyze false negatives and false positives
3. analyze_correct_predictions.py - Analyze true positives and true negatives
4. comparative_analysis.py - Compare wrong vs correct predictions
5. time_diff_comparison_with_ratio.py - Time difference comparison with ratio line
6. scenario_distribution_analysis.py - Scenario distribution and performance analysis

Usage:
    python prediction_analysis/scripts/run_all_analyses.py

Output structure:
    prediction_analysis/
    ├── scripts/                   - Analysis scripts
    ├── wrong_predictions/         - Wrong prediction data and visualizations
    ├── correct_predictions/       - Correct prediction data and visualizations
    ├── comparative_analysis/      - Comparative visualizations and statistics
    └── scenario_distribution/     - Scenario distribution and performance analysis
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
        ("time_diff_comparison_with_ratio.py", "Time difference comparison with ratio line"),
        ("scenario_distribution_analysis.py", "Scenario distribution and performance analysis"),
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
│   ├── comparative_analysis.py
│   └── time_diff_comparison_with_ratio.py  (with ratio line)
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
    ├── comparison_overlapping_histograms.png/pdf  (⭐ with ratio line)
    ├── comparison_overlapping_histograms_normalized.png/pdf
    ├── comparison_histograms.png/pdf
    ├── comparison_combined_boxplot.png/pdf
    ├── comparison_scatter_plots.png/pdf
    ├── comparison_violin_plot.png/pdf
    ├── correct_predictions_analysis.png/pdf
    ├── time_diff_comparison_histograms.png/pdf
    ├── time_diff_comparison_boxplots.png/pdf
    ├── time_diff_comparison_violin.png/pdf
    ├── confidence_comparison.png/pdf
    └── statistical_comparison.txt
└── scenario_distribution/
    ├── dataset_distribution_level1.png/pdf      (scenario counts & pie chart)
    ├── dataset_distribution_level2.png/pdf
    ├── prediction_outcomes_level1.png/pdf       (TP/TN/FP/FN per scenario)
    ├── prediction_outcomes_level2.png/pdf
    ├── error_rate_level1.png/pdf                (error rate per scenario)
    ├── error_rate_level2.png/pdf
    ├── representation_comparison_level1.png/pdf (dataset vs errors)
    ├── representation_comparison_level2.png/pdf
    ├── scenario_heatmap_level1.png/pdf          (outcome × scenario matrix)
    ├── scenario_heatmap_level2.png/pdf
    ├── false_negatives_level1.png/pdf           (⚠️ missed accidents analysis)
    ├── false_negatives_level2.png/pdf
    └── scenario_statistics.txt
""")


if __name__ == '__main__':
    main()
