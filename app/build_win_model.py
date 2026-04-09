"""
Train a GradientBoosting win probability model from matchup features.

Input:  data/Xy_train.csv (produced by data_pipeline/build_matchup_features.py)
Output: data/production_model.pkl
"""

import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

FEATURES = [
    "QB_diff", "OL_adv_1", "OL_adv_2", "WR_adv_1", "WR_adv_2",
    "RB_adv_1", "RB_adv_2", "TE_diff", "PK_diff", "P_diff",
    "QB1_vs_DB_2", "QB2_vs_DB_1", "PASS_SYNERGY_diff", "team_1_is_home",
]


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")

    Xy_train = pd.read_csv(os.path.join(data_dir, "Xy_train.csv"))
    X_train = Xy_train[FEATURES]
    y_train = Xy_train["target"]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            subsample=0.8,
            n_estimators=100,
            min_samples_split=5,
            min_samples_leaf=4,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )),
    ])

    pipe.fit(X_train, y_train)

    out_path = os.path.join(data_dir, "production_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
