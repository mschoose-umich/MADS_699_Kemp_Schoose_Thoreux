# Data Leakage Assessment — Power Rankings Models

## Overview

This document assesses data leakage in two prediction models:
1. **LightGBM** in `generate_power_rankings.py` (walk-forward + multi-year chained predictions)
2. **Ridge Regression** in `power_rankings.ipynb` (cells 20-22, 2026 projection)

Plus a **weighted composite** (notebook cells 12-13) and an **in-sample Ridge** (cell 14).

---

## Legitimate (Not Leakage)

**Using Year Y SP+ metrics to predict Year Y+1 SP+ rating** is temporally valid:
- `rating`, `off_rating`, `def_rating`, sub-metrics, `sos`, `st_rating` are all end-of-season Year Y metrics
- `prior_year_rating` (Year Y-1) is also safe
- Roster recruiting features (star counts) reflect historical high school evaluations
- `talent` composite is a pre-season metric based on recruiting classes

**Verdict**: The core train/predict setup (Year Y features → Year Y+1 target) is NOT leaking.

---

## Actual Issues Found

### Issue A: Walk-Forward Composite Score Mismatch
- **Severity**: Medium (conceptual, not statistical leakage)
- **Location**: `generate_power_rankings.py:118-197`
- **Problem**: Model trains on "Year Y features → Year Y+1 rating" but outputs are labeled as Year Y's `composite_score`. The composite score for Year Y is actually the model's prediction of Year Y+1, mislabeled as current-year ranking.
- **Impact**: Historical rankings (2016-2025) represent predicted next-year performance, not current-year performance.
- **Fix**: Renamed to `projected_next_year_rating` and added documentation clarifying the semantics.

### Issue B: Cell 14 Ridge — In-Sample Circularity
- **Severity**: High (methodologically invalid)
- **Location**: `power_rankings.ipynb` cell 14
- **Problem**: Trains Ridge on ALL years to predict `rating` from sub-metrics with no train/test split. Reports "in-sample R²" which is meaningless for generalization. Features (`off_rating`, `def_rating`, sub-metrics) are components of `rating` itself — this is tautological.
- **Fix**: Added temporal train/test split (train ≤2023, test 2024-2025) with cross-validation.

### Issue C: TimeSeriesSplit Without Proper Year Grouping
- **Severity**: Medium (leaks within-year distributional info)
- **Location**: `generate_power_rankings.py:269`
- **Problem**: `TimeSeriesSplit(n_splits=4)` splits by row index. Data sorted by `["team", "year"]` means splits don't cleanly separate years — a fold could put Ohio State 2020 in train and Michigan 2020 in validation.
- **Fix**: Replaced with custom year-based GroupTimeSeriesSplit that ensures all teams from the same year stay together.

### Issue D: StandardScaler Fit Before CV
- **Severity**: Low (minor leakage)
- **Location**: `generate_power_rankings.py:253-256`
- **Problem**: Scaler `fit_transform`'d on entire `X_train` before `RandomizedSearchCV`. CV validation folds see scaled data where scaler was fitted including their data.
- **Fix**: Wrapped `StandardScaler` + `LGBMRegressor` in a `sklearn.pipeline.Pipeline` inside CV.

### Issue E: No Baseline Comparison
- **Severity**: Medium (can't assess model value)
- **Problem**: Neither model compares against a naive baseline. SP+ ratings are highly autocorrelated — persistence ("next year = this year") likely achieves high R².
- **Fix**: Added persistence baseline comparison reporting RMSE/R² alongside model metrics.

---

## Summary of Fixes Applied

| Issue | File | Fix |
|-------|------|-----|
| A. Composite mislabel | `generate_power_rankings.py` | Renamed column, added docstring clarification |
| B. In-sample Ridge | `power_rankings.ipynb` cell 14 | Added temporal train/test split with CV |
| C. TimeSeriesSplit | `generate_power_rankings.py` | Year-based grouped CV splits |
| D. Scaler in CV | `generate_power_rankings.py` | Pipeline wrapping scaler + model |
| E. No baseline | `generate_power_rankings.py` | Persistence baseline RMSE/R² reporting |
