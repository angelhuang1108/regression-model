# Clinical Trial Duration Model — Interview Walkthrough

## Overview

This project predicts clinical trial duration (start → primary completion) using structured features derived from ClinicalTrials.gov data. Models are trained per phase using gradient boosting and evaluated on held-out data.

## Pipeline (high level)

1. Build modeling cohort → `step00_cohort_io.py`
2. Construct features → `step01_features.py`
3. Define targets → `step02_targets.py`
4. Train models → `step03_train_regression.py`
5. Evaluate results → `step04_evaluation.py`

## How to run

```bash
python 4_regression/core/step03_train_regression.py
```

## Key output

* `6_results/regression_report.txt`

## Suggested walkthrough order

1. `step00_cohort_io.py` — data assembly
2. `step01_features.py` — feature construction
3. `step02_targets.py` — target definitions
4. `step03_train_regression.py` — training logic
5. `step04_evaluation.py` — metrics
6. `6_results/regression_report.txt` — results

## Notes

* Models are trained per phase to account for structural differences
* HistGradientBoosting is used for non-linear modeling with native NaN handling
* Features reflect trial design and operational complexity
