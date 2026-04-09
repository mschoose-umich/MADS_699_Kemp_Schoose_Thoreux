import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def main():
    # 1. Load Data
    adv = pd.read_csv(os.path.join(DATA_DIR, 'advanced_metrics.csv'))
    
    # 2. Setup Variables
    # Rating = SP+ rating (predicted ranking proxy)
    # Ranking = actual SP+ ranking
    df = adv.copy().sort_values(['team', 'year'])
    df['prior_year_rating'] = df.groupby('team')['rating'].shift(1)
    df['prior_year_ranking'] = df.groupby('team')['ranking'].shift(1)
    
    NIL_YEAR = 2021
    df['post_nil'] = (df['year'] >= NIL_YEAR).astype(int)
    
    # Classify blue-chip based on 2015-2019 baseline talent
    pre_nil_talent = df[df['year'] <= 2019].groupby('team')['talent'].mean()
    talent_median = pre_nil_talent.median()
    blue_chip_teams = set(pre_nil_talent[pre_nil_talent >= talent_median].index)
    df['blue_chip'] = df['team'].isin(blue_chip_teams).astype(int)
    
    # Drop rows with missing values for analysis
    df_clean = df.dropna(subset=['rating', 'ranking', 'talent', 'prior_year_rating']).copy()
    
    print("="*60)
    print("ANOVA: Variation in Predicted Rankings (SP+ Rating) by NIL era and Blue-Chip Status")
    print("="*60)
    # Two-way ANOVA without covariate
    # Formula: rating ~ C(post_nil) + C(blue_chip) + C(post_nil):C(blue_chip)
    model_anova_rating = ols('rating ~ C(post_nil) * C(blue_chip)', data=df_clean).fit()
    anova_table_rating = sm.stats.anova_lm(model_anova_rating, typ=2)
    print(anova_table_rating)
    print("\n")
    
    print("="*60)
    print("ANOVA: Variation in Rankings (National Rank) by NIL era and Blue-Chip Status")
    print("="*60)
    model_anova_ranking = ols('ranking ~ C(post_nil) * C(blue_chip)', data=df_clean).fit()
    anova_table_ranking = sm.stats.anova_lm(model_anova_ranking, typ=2)
    print(anova_table_ranking)
    print("\n")
    
    print("="*60)
    print("ANCOVA: Predicted Rankings (SP+ Rating) controlling for Prior Year Rating and Talent")
    print("="*60)
    # ANCOVA with covariates (prior_year_rating, talent)
    model_ancova_rating = ols('rating ~ C(post_nil) * C(blue_chip) + prior_year_rating + talent', data=df_clean).fit()
    ancova_table_rating = sm.stats.anova_lm(model_ancova_rating, typ=2)
    print(ancova_table_rating)
    print("\n")
    
    print("="*60)
    print("ANCOVA: Rankings (National Rank) controlling for Prior Year Ranking and Talent")
    print("="*60)
    model_ancova_ranking = ols('ranking ~ C(post_nil) * C(blue_chip) + prior_year_ranking + talent', data=df_clean).fit()
    ancova_table_ranking = sm.stats.anova_lm(model_ancova_ranking, typ=2)
    print(ancova_table_ranking)
    print("\n")

if __name__ == "__main__":
    main()
