# Scenario Analysis - Two Versions

You now have **TWO versions** of scenario analysis:

## 1. analyze_scenarios_VAL.py (Validation Only)

**Purpose**: Unbiased model performance evaluation

```bash
python analyze_scenarios_VAL.py
```

**What it does:**
- Uses **validation set only** (~130 videos)
- Compares model predictions vs ground truth
- Calculates **accuracy** per scenario
- ✓ Unbiased: Model never saw this data during training
- ✗ Small sample: Some scenarios have few examples

**Output**: `scenario_analysis/` directory
- CSV files: Accuracy metrics per scenario
- PNG plots: Accuracy visualization

**Use this for**: Model evaluation, performance reports

---

## 2. analyze_scenarios_ALL.py (Full Dataset)

**Purpose**: Comprehensive scenario distribution analysis

```bash
python analyze_scenarios_ALL.py
```

**What it does:**
- Uses **entire dataset** (~640 videos for Round 1, ~560 for Round 2)
- Shows scenario distribution (accident rate per scenario)
- ✓ Large sample: 4-5x more data (much more reliable statistics)
- ✓ Better coverage: More examples per scenario
- ✗ Includes training data: Don't use for model evaluation

**Output**: `scenario_analysis_ALL/` directory
- CSV files: Distribution statistics (accident rate, counts)
- PNG plots: Distribution visualization

**Use this for**: Understanding scenario patterns, annotation analysis, scenario coverage

---

## Sample Size Comparison

| Dataset | VAL (validation) | ALL (full) | Increase |
|---------|------------------|------------|----------|
| Round 1 | 131 videos | 642 videos | **+390%** |
| Round 2 | 110 videos | 559 videos | **+408%** |
| Level 2 matched | 67 videos | 336 videos | **+401%** |

---

## When to Use Which?

### Use VAL version when:
- ✓ Evaluating model performance
- ✓ Comparing different models
- ✓ Reporting accuracy metrics
- ✓ Need unbiased results

### Use ALL version when:
- ✓ Analyzing scenario distribution
- ✓ Understanding accident patterns
- ✓ Checking annotation coverage
- ✓ Need statistically reliable counts
- ✓ Planning data collection

---

## Key Differences

| Feature | VAL | ALL |
|---------|-----|-----|
| Data | Validation only | Train + Val |
| Sample size | ~130 videos | ~640 videos |
| Metric | Accuracy (%) | Accident rate (%) |
| Model predictions | Yes | No (ground truth only) |
| Biased? | No ✓ | Yes (includes training data) |
| Use for evaluation? | Yes ✓ | No ✗ |
| Use for statistics? | Limited | Yes ✓ |

---

## Quick Start

```bash
# For model evaluation (unbiased)
python analyze_scenarios_VAL.py

# For scenario statistics (large sample)
python analyze_scenarios_ALL.py
```

Both scripts are independent and can be run separately!
