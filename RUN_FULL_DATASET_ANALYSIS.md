# Full Dataset Scenario Analysis - Quick Guide

## Overview
This guide explains how to run scenario analysis on the **ENTIRE dataset** (train + validation) to get better per-scenario statistics with 4-5x more samples.

⚠️ **IMPORTANT WARNING**: This analysis includes training data, so accuracy will be **BIASED/INFLATED**. Use this ONLY for:
- Better per-scenario statistics (640 videos vs 130 in validation)
- Understanding which scenarios the model struggles with most
- Getting more reliable sample sizes per scenario

For **unbiased evaluation**, use `analyze_scenarios_VAL.py` instead.

## Two-Step Process

### Step 1: Generate Predictions on Full Dataset
Run model predictions on all 1500 videos (train + val):

```bash
python predict_all_dataset.py
```

**What it does:**
- Loads trained model from `logs/archive/best_videomae_large_model.pth`
- Processes `balanced_dataset_2s/train/` (train split)
- Processes `balanced_dataset_2s/val/` (validation split)
- Combines predictions from both splits (~1500 videos)
- Aggregates 2s segment predictions to video-level using 'any' method
- Outputs to: `aggregated_results_ALL/aggregated_predictions_any.csv`

**Expected output:**
```
Processing train split...
  positive/: XXXX segments
  negative/: XXXX segments
Processing val split...
  positive/: XXXX segments
  negative/: XXXX segments
Combined: ~1500 video predictions
Aggregated to video level: 1500 videos
```

**Runtime:** ~5-10 minutes on GPU

---

### Step 2: Analyze by Scenario
Run scenario analysis using the predictions:

```bash
python analyze_scenarios_ALL.py
```

**What it does:**
- Loads scenario annotations (Round 1: 642 videos, Round 2: 559 videos)
- Matches annotations between rounds
- Merges with model predictions from Step 1
- Calculates per-scenario accuracy (Level 1 and Level 2)
- Generates visualization plots

**Outputs to:** `scenario_analysis_ALL/`

**Files created:**
```
CSV files (13 total):
  - Level_1_matched.csv, Level_2_matched.csv (annotation matching)
  - round_1_prediction.csv, round_2_prediction.csv (predictions with annotations)
  - *_level1_accuracy.csv (per-scenario accuracy, Level 1)
  - *_level2_accuracy.csv (hierarchical Level 2 accuracy)

PNG plots (7 total):
  - round_1_level1_accuracy.png
  - round_2_level1_accuracy.png
  - level_1_matched_level1_accuracy.png
  - round_1_level2_accuracy.png
  - round_2_level2_accuracy.png
  - level_2_matched_level2_accuracy.png
```

---

## Expected Results

### Sample Size Comparison
| Dataset | Videos | Notes |
|---------|--------|-------|
| Validation only | ~130 | Unbiased, small samples |
| Full dataset | ~640 | Biased, 4-5x larger samples |

### Typical Output (Round 1, Level 1):
```
scenario_level_1 Accuracy (BIASED - includes training data):
================================================================================
Front collision              : 340/350 = 97.1%  (avg prob: 0.958)
Rear-end collision           : 145/150 = 96.7%  (avg prob: 0.943)
Side collision               :  78/82  = 95.1%  (avg prob: 0.921)
...
```

⚠️ **Note**: Accuracy will be 5-10% higher than validation-only results due to training data leakage!

---

## Key Differences: VAL vs ALL

| Aspect | analyze_scenarios_VAL.py | analyze_scenarios_ALL.py |
|--------|-------------------------|--------------------------|
| **Data** | Validation only (293 videos) | Full dataset (1500 videos) |
| **Annotated videos** | ~130 | ~640 |
| **Accuracy** | Unbiased (~91%) | Biased/Inflated (~96%) |
| **Sample size** | Small, less reliable | Large, more reliable |
| **Use for** | Final evaluation, reporting | Scenario statistics |
| **Training data** | ✗ Excluded | ✓ Included |

---

## When to Use Each

### Use `analyze_scenarios_VAL.py` (Validation only):
✓ Final model evaluation  
✓ Reporting accuracy to stakeholders  
✓ Comparing model versions  
✓ Unbiased performance metrics  

### Use `analyze_scenarios_ALL.py` (Full dataset):
✓ Better per-scenario statistics (4-5x more data)  
✓ Identifying which scenarios need more work  
✓ Understanding model behavior across all scenarios  
✓ When you need reliable sample sizes per scenario  

---

## Quick Commands

```bash
# Full dataset analysis (2 steps)
python predict_all_dataset.py
python analyze_scenarios_ALL.py

# Validation-only analysis (1 step)
python analyze_scenarios_VAL.py
```

---

## Troubleshooting

**Error: "File not found: aggregated_results_ALL/aggregated_predictions_any.csv"**
- Solution: Run Step 1 first (`python predict_all_dataset.py`)

**Error: "CUDA out of memory"**
- Solution: Reduce batch size in `predict_all_dataset.py` (line ~50: `batch_size=8`)

**Predictions taking too long:**
- Check GPU availability: `nvidia-smi`
- Model runs on CPU if GPU unavailable (much slower)

---

## Summary

1. **Step 1**: `predict_all_dataset.py` → Generate predictions on full dataset
2. **Step 2**: `analyze_scenarios_ALL.py` → Analyze by scenario
3. **Remember**: Results are **BIASED** due to training data inclusion
4. **Use VAL version** for unbiased evaluation: `analyze_scenarios_VAL.py`

**Bottom line:** ALL version gives you 4-5x more data per scenario, but accuracy is inflated. VAL version gives you unbiased metrics but smaller samples.
