# Project Assessment & Improvement Plan
*Generated: 2026-03-17 | Updated with Book-Based Causal Critique*

---

## Current State Baseline
- **Model**: LightGBM predicting next-year SP+ rating (24 features)
- **Metrics**: R²=0.616, RMSE=6.73, Spearman r=0.823
- **NIL analysis**: ITS + DiD with two-way FEs (team+year), panel ITS, parallel trends test, sensitivity analysis

---

## Causal Inference Refinement (High Priority - Book Based)

### 1. Spec Correction (Collinearity)
- [x] **Fix TWFE Spec**: In `power_rankings.ipynb` and `nil_anova_ancova.py`, remove `post_nil` and `blue_chip` main effects when using Year and Team Fixed Effects. Document that they are absorbed by the FEs.
- [x] **Update Coefficient Interpretation**: Ensure the only reported coefficient for the DiD model is the interaction term `post_nil_x_blue_chip`.

### 2. Validation & Falsification
- [x] **Time-Placebo Test**: Implement a full DiD re-run with a fake treatment date (e.g., 2018) using only pre-NIL data (2015-2020). Verify the coefficient is indistinguishable from zero.
- [x] **Functional Form Sensitivity**: Re-run DiD analysis using `log(rating)` and `log(talent)` to ensure parallel trends and treatment effects are not scale-dependent.
- [x] **SUTVA Discussion**: Add a markdown section to the notebook/report explicitly discussing the zero-sum nature of CFB recruiting as a SUTVA violation.

### 3. Advanced Estimators (Optional/Low Priority)
- [ ] **Synthetic DiD (SDID)**: Investigate `synthdid` or `CausalPy` for a more robust estimator given the single treatment date and fixed number of units.
- [ ] **Aggregate ITS Assessment**: Relabel aggregate ITS (11 observations) as "illustrative" or drop in favor of the more powerful panel-level analyses.

---

## Feature & Model Improvements (Ongoing)

### 1. Feature Gaps
- [x] **Coach tenure/first-year flag**: Implemented.
- [ ] **Returning starters %**: Pending data source.
- [x] **Transfer portal net flows**: Data fetched, pending integration for post-2021 years.
- [ ] **Position-specific recruiting**: Data fetched, pending coverage improvement.

### 2. Rankings Methodology
- [x] **Bradley-Terry Hybrid**: Implemented as `prior_bt_strength` feature.

---

## Revised Priority Table (Causal Analysis)

| Priority | Issue | Book Reference | Fix |
| :--- | :--- | :--- | :--- |
| **High** | Collinear post_nil/blue_chip with TWFE | All books (TWFE spec) | Remove or document as absorbed |
| **High** | No time-placebo falsification test | CI in Python ch08 | Run DiD with fake 2018 cutoff on pre-data |
| **High** | SUTVA violation undiscussed | CI in Python ch08 | Add limitation section acknowledging zero-sum talent market |
| **Medium** | Functional form sensitivity | CI in Python ch08 | Test with log-transformed outcomes |
| **Medium** | No DiD in pipeline/dashboard | — | Port results to generate_power_rankings.py |
| **Low** | Consider SDID estimator | CI in Python ch09 | Implement via CausalPy or synthdid |
| **Low** | Aggregate ITS underpowered (11 obs) | — | Label as illustrative or drop |

---

## Implementation Notes
- **TWFE Spec**: Per *Causal Inference in Python*, the model is $Y_{it} = \tau W_{it} + \alpha_i + \gamma_t + e_{it}$.
- **SUTVA**: CFB recruiting is a textbook case of spillover; treatment of one unit (elite team getting a recruit) affects the control group's potential.
- **Scale Invariance**: Parallel trends in levels $\neq$ parallel trends in logs.
