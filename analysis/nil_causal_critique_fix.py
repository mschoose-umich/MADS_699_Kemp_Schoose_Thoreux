import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def run_did_analysis(df, outcome_col, treatment_year, group_col='blue_chip', log_transform=False, title_prefix=""):
    """
    Runs a Two-Way Fixed Effects (TWFE) DiD analysis.
    Per book critique: Main effects (post_nil, blue_chip) are absorbed by Year and Team FEs.
    Model: Y ~ interaction + C(year) + C(team)
    """
    df_clean = df.dropna(subset=[outcome_col, 'talent']).copy()
    
    if log_transform:
        # Avoid log(0)
        df_clean[f'log_{outcome_col}'] = np.log(df_clean[outcome_col] + 1e-6)
        outcome = f'log_{outcome_col}'
    else:
        outcome = outcome_col
        
    df_clean['post_treatment'] = (df_clean['year'] >= treatment_year).astype(int)
    df_clean['interaction'] = df_clean['post_treatment'] * df_clean[group_col]
    
    # Formula approach for TWFE
    # We omit post_treatment and blue_chip as they are collinear with C(year) and C(team)
    formula = f'{outcome} ~ interaction + C(year) + C(team)'
    
    # Cluster standard errors by team
    model = ols(formula, data=df_clean).fit(cov_type='cluster', cov_kwds={'groups': df_clean['team'].values})
    
    return model

def main():
    # 1. Load Data
    adv = pd.read_csv(os.path.join(DATA_DIR, 'advanced_metrics.csv'))
    df = adv.copy().sort_values(['team', 'year'])
    
    # Baseline blue-chip classification (2015-2019 window per original notebook logic)
    pre_nil_talent = df[df['year'] <= 2019].groupby('team')['talent'].mean()
    talent_median = pre_nil_talent.median()
    blue_chip_teams = set(pre_nil_talent[pre_nil_talent >= talent_median].index)
    df['blue_chip'] = df['team'].isin(blue_chip_teams).astype(int)
    
    print("="*80)
    print("REFINED CAUSAL ANALYSIS: NIL IMPACT ON COMPETITIVE LANDSCAPE")
    print("Addressing Book-Based Critiques: TWFE Spec, Placebo, and Scale Sensitivity")
    print("="*80)
    
    NIL_YEAR = 2021
    outcomes = ['rating', 'talent']
    
    # --- 1. TWFE Spec Correction ---
    print("\n[1] TWFE SPEC CORRECTION (Absorbing main effects into FEs)")
    for outcome in outcomes:
        model = run_did_analysis(df, outcome, NIL_YEAR)
        coef = model.params['interaction']
        p = model.pvalues['interaction']
        ci = model.conf_int().loc['interaction']
        print(f"Outcome: {outcome:8} | DiD Coef: {coef:8.4f} | p: {p:.4f} | 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # --- 2. Time-Placebo Falsification Test ---
    print("\n[2] TIME-PLACEBO TEST (Fake NIL Breakpoint at 2018, Pre-period only)")
    # Filter to pre-NIL period only (2015-2020)
    df_pre = df[df['year'] <= 2020].copy()
    PLACEBO_YEAR = 2018
    for outcome in outcomes:
        model = run_did_analysis(df_pre, outcome, PLACEBO_YEAR)
        coef = model.params['interaction']
        p = model.pvalues['interaction']
        result = "PASS ✓" if p > 0.10 else "FAIL ✗ (Spurious pre-trend detected)"
        print(f"Outcome: {outcome:8} | Placebo Coef: {coef:8.4f} | p: {p:.4f} | Result: {result}")

    # --- 3. Functional Form Sensitivity (Log Transform) ---
    print("\n[3] FUNCTIONAL FORM SENSITIVITY (Log-transformed outcomes)")
    for outcome in outcomes:
        # Rating can be negative (SP+), so we handle that if necessary. 
        # Talent is always positive.
        if outcome == 'rating':
            # Shift ratings to be positive for log transform
            min_rating = df['rating'].min()
            df[f'adj_{outcome}'] = df['rating'] - min_rating + 1
            outcome_to_run = f'adj_{outcome}'
        else:
            outcome_to_run = outcome
            
        model = run_did_analysis(df, outcome_to_run, NIL_YEAR, log_transform=True)
        coef = model.params['interaction']
        p = model.pvalues['interaction']
        print(f"Outcome: log({outcome:8}) | DiD Coef: {coef:8.4f} | p: {p:.4f}")

    # --- 4. SUTVA Violation Acknowledgement ---
    print("\n" + "-"*80)
    print("LIMITATION: SUTVA VIOLATION (Unit Spillover)")
    print("College football recruiting is a zero-sum market. If 'blue-chip' programs gain")
    print("a 5-star recruit, that player is removed from the potential pool for all other")
    print("teams. This violates the Stable Unit Treatment Value Assumption (SUTVA).")
    print("The estimated treatment effect represents the RELATIVE advantage shift in this")
    print("competitive equilibrium, not an absolute performance gain in isolation.")
    print("-"*80)

if __name__ == "__main__":
    main()
