#!/bin/bash
# Quick script to run scenario analysis
# Works with any annotation files - automatically adapts to whatever scenarios exist

echo "Running scenario analysis..."
echo "This will:"
echo "  1. Match annotations between round_1.csv and round_2.csv"
echo "  2. Merge with predictions from aggregated_results/"
echo "  3. Calculate per-scenario accuracy for all existing scenarios"
echo ""

python analyze_scenarios.py

echo ""
echo "Done! Results saved to scenario_analysis/"
echo ""
echo "Key outputs:"
echo "  - round_1_prediction.csv, round_2_prediction.csv: All predictions with scenario labels"
echo "  - level_2_matched_prediction.csv: Only videos where both annotators fully agree"
echo "  - *_level1_accuracy.csv: Accuracy by high-level scenario"
echo "  - *_level2_accuracy.csv: Accuracy by detailed sub-scenario (with parent Level 1)"
