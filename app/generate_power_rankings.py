"""
Generate predictive power rankings for college football teams.

Loads SP+ ratings, talent composite, and optionally player-level recruiting
data to compute a weighted composite score. Outputs national and conference
rankings as CSVs.

Prerequisites:
    python get_recruit_data.py
    python get_roster_data.py
    python merge_roster_rankings.py
    python get_advanced_metrics.py
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

import config

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def estimate_decay_rate(df):
    """Estimate AR(1) mean-reversion coefficient from historical SP+ ratings.

    Fits rating_t = alpha + beta * rating_{t-1} on all team-year pairs.
    Returns beta as the empirical decay rate (replaces hardcoded config value).
    """
    df_sorted = df.sort_values(["team", "year"]).copy()
    df_sorted["prior_rating"] = df_sorted.groupby("team")["rating"].shift(1)
    pairs = df_sorted.dropna(subset=["prior_rating", "rating"])

    x = pairs["prior_rating"].values
    y = pairs["rating"].values
    beta = float(np.cov(x, y)[0, 1] / np.var(x))
    print(f"  Empirical AR(1) decay rate: {beta:.4f} (config fallback: {config.decay_rate})")
    return beta


class YearBasedTimeSeriesSplit:
    """Cross-validator that splits by year groups, ensuring no same-year leakage.

    For each split, all data from earlier years goes to train and all data
    from a later year goes to validation. This prevents within-year
    distributional information from leaking across folds.
    """

    def __init__(self, n_splits=4):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (year values) must be provided")
        unique_years = sorted(set(groups))
        n_years = len(unique_years)
        if n_years < self.n_splits + 1:
            raise ValueError(
                f"Need at least {self.n_splits + 1} years, got {n_years}"
            )
        # Use the last n_splits years as validation folds
        for i in range(self.n_splits):
            val_year_idx = n_years - self.n_splits + i
            val_year = unique_years[val_year_idx]
            train_years = set(unique_years[:val_year_idx])

            train_idx = [j for j, g in enumerate(groups) if g in train_years]
            val_idx = [j for j, g in enumerate(groups) if g == val_year]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


POSITION_GROUPS = {
    "QB": ["QB"],
    "Skill": ["RB", "FB", "WR", "TE"],
    "OL": ["OL", "G", "OT", "C"],
    "DL": ["DL", "DE", "DT", "NT"],
    "LB": ["LB"],
    "DB": ["DB", "CB", "S"],
}

# Invert for fast lookup: position -> group
_POS_TO_GROUP = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _grp


def parse_dict_column(df, col, prefix):
    """Parse a stringified dict column into flat columns with a prefix."""
    parsed = df[col].apply(ast.literal_eval)
    normalized = pd.json_normalize(parsed)
    normalized.columns = [f"{prefix}_{c}" for c in normalized.columns]
    return normalized


def load_advanced_metrics(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "advanced_metrics.csv")
    """Load and flatten the advanced metrics CSV."""
    adv = pd.read_csv(path)

    off_cols = parse_dict_column(adv, "offense", "off")
    def_cols = parse_dict_column(adv, "defense", "def")
    st_cols = parse_dict_column(adv, "specialTeams", "st")

    return pd.concat(
        [adv.drop(columns=["offense", "defense", "specialTeams"]),
         off_cols, def_cols, st_cols],
        axis=1,
    )


def load_roster_aggregates(roster_dir=None):
    if roster_dir is None:
        roster_dir = os.path.join(DATA_DIR, "merged_rosters")
    """Aggregate player-level recruiting data per team per year.

    Returns ~20 features per team-year: star-level counts, rated player
    counts/percentages, and per-position-group elite/rated counts.
    """
    roster_path = Path(roster_dir)
    roster_files = sorted(roster_path.glob("*.csv")) if roster_path.exists() else []

    if not roster_files:
        return pd.DataFrame()

    records = []
    for f in roster_files:
        rdf = pd.read_csv(f)
        # Extract year from filename (e.g., 2020_rosters.csv, roster_2020.csv, 2020.csv)
        year_match = re.search(r"(\d{4})", f.stem)
        year = int(year_match.group(1))

        for team, grp in rdf.groupby("team"):
            rated = grp["stars"].dropna()
            roster_size = len(grp)
            rec = {
                "year": year,
                "team": team,
                "star_5_count": int((rated == 5).sum()),
                "star_4_count": int((rated == 4).sum()),
                "star_3_count": int((rated == 3).sum()),
                "star_2_count": int((rated == 2).sum()),
                "rated_player_count": len(rated),
                "rated_player_pct": len(rated) / roster_size if roster_size > 0 else 0,
            }

            # Position-group features
            pos_col = "position" if "position" in grp.columns else None
            for pg in POSITION_GROUPS:
                if pos_col:
                    pg_mask = grp[pos_col].isin(POSITION_GROUPS[pg])
                    pg_stars = grp.loc[pg_mask, "stars"].dropna()
                    rec[f"{pg}_elite_count"] = int((pg_stars >= 4).sum())
                    rec[f"{pg}_rated_count"] = int(len(pg_stars))
                else:
                    rec[f"{pg}_elite_count"] = 0
                    rec[f"{pg}_rated_count"] = 0

            records.append(rec)

    return pd.DataFrame(records)


def compute_rankings(df_flat, roster_agg, best_params,
                     ranker=None, ranker_scaler=None, pred_features_override=None):
    """Compute power rankings using walk-forward model scoring.

    For each year Y (2016-2025), trains LightGBM on all data from years < Y
    to predict next-year SP+ rating. The predicted_rating for year Y is the
    model's forecast of Year Y+1 performance (i.e., a forward-looking metric).
    Year 2015 uses raw SP+ rating as a fallback.

    If a trained LGBMRanker + scaler are provided, final rank columns are
    derived from ranker scores (better Spearman r); otherwise from
    predicted_rating.
    """
    df = df_flat.copy()

    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")

    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)

    if pred_features_override is not None:
        pred_features = [c for c in pred_features_override if c in df.columns]
    else:
        pred_features = _build_feature_list(df[(df["year"] >= 2016)])

    all_years = sorted(df["year"].unique())
    scored_frames = []

    for year in all_years:
        year_df = df[df["year"] == year].copy()

        if year == all_years[0]:
            # First year: no prior data to train on, use raw SP+ rating
            year_df["predicted_rating"] = year_df["rating"].round(2)
        else:
            # Walk-forward: train on all prior years
            train_df = df[df["year"] < year].dropna(subset=["rating"]).copy()
            train_df["next_year_rating"] = train_df.groupby("team")["rating"].shift(-1)
            train_df = train_df.dropna(subset=["next_year_rating"])
            train_medians_wf = train_df[pred_features].median(numeric_only=True)
            train_df = fill_features(train_df, pred_features, train_medians_wf)

            if len(train_df) < 20:
                year_df["predicted_rating"] = year_df["rating"].round(2)
            else:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    scaler.fit_transform(train_df[pred_features]),
                    columns=pred_features, index=train_df.index,
                )
                y_tr = train_df["next_year_rating"].values

                wf_model = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=42)
                wf_model.fit(X_tr, y_tr)

                # Predict current year — impute with training medians to keep G5 teams
                year_filled = fill_features(year_df, pred_features, train_medians_wf)
                X_pred = pd.DataFrame(
                    scaler.transform(year_filled[pred_features]),
                    columns=pred_features, index=year_filled.index,
                )
                year_df["predicted_rating"] = wf_model.predict(X_pred).round(2)
                year_df["predicted_rating"] = year_df["predicted_rating"].fillna(
                    year_df["rating"].round(2)
                )

        scored_frames.append(year_df)

    df = pd.concat(scored_frames, ignore_index=True)

    # Rank: prefer ranker scores (ordinal model) when available
    if ranker is not None and ranker_scaler is not None:
        feature_medians = df[pred_features].median(numeric_only=True)
        rank_signal = pd.Series(np.nan, index=df.index)
        for yr, grp in df.groupby("year"):
            if grp[pred_features].dropna(how="all").empty:
                continue
            scores = score_with_ranker(
                ranker, ranker_scaler, grp, pred_features, feature_medians
            )
            rank_signal.loc[grp.index] = scores
        df["ranker_score"] = rank_signal.round(4)
        # For years before the ranker's training horizon, fall back to predicted_rating
        df["ranker_score"] = df["ranker_score"].fillna(df["predicted_rating"])
        rank_col = "ranker_score"
    else:
        rank_col = "predicted_rating"

    df["national_rank"] = (
        df.groupby("year")[rank_col]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    df["conference_rank"] = (
        df.groupby(["year", "conference"])[rank_col]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    return df


def _build_feature_list(df):
    """Build the list of predictor features from available columns.

    Includes SP+ metrics, sub-metrics, roster features, BT strength,
    coaching continuity, and transfer portal features.
    Drops features with >50% NaN in the provided dataframe.
    """
    sp_features = [
        "rating", "off_rating", "def_rating", "talent", "prior_year_rating",
        "off_success", "def_success", "off_explosiveness", "def_explosiveness",
        "def_havoc.total", "st_rating", "sos",
    ]
    contextual_features = [
        # Coaching continuity — new head coaches cause multi-year adjustment periods
        "coach_tenure", "is_first_year_coach",
        # Transfer portal net flows (coverage: 2021+; dropped if >50% NaN in training set)
        "portal_net_count", "portal_net_rating", "portal_inbound_count",
        # Position-group continuous recruiting quality (replaces coarse star bins)
        "rec_qb", "rec_rb", "rec_wr", "rec_ol", "rec_dl", "rec_lb", "rec_db",
        # BT prior-year realized game strength (available but not included by default;
        # current-year BT is collinear with SP+, lagged BT marginal — kept for research)
        # "prior_bt_strength",
    ]
    roster_features = [
        "star_5_count", "star_4_count", "star_3_count", "star_2_count",
        "rated_player_count", "rated_player_pct",
    ]
    for pg in POSITION_GROUPS:
        roster_features += [f"{pg}_elite_count", f"{pg}_rated_count"]

    candidates = sp_features + contextual_features + roster_features
    available = [c for c in candidates if c in df.columns]

    # Drop features with >50% NaN
    threshold = 0.5 * len(df)
    return [c for c in available if df[c].notna().sum() > threshold]


def train_lgbm_model(df_flat, roster_agg):
    """Train a LightGBM model with RandomizedSearchCV to predict next-year SP+ rating.

    Returns: best model, scaler, residual_std, pred_features list, best_params dict
    """
    df = df_flat.copy()

    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")

    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)

    # Target: next-year SP+ rating
    df["next_year_rating"] = df.groupby("team")["rating"].shift(-1)

    # Training set: 2016-2024 (need prior_year_rating, so skip 2015)
    train = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()

    pred_features = _build_feature_list(train)
    print(f"  Using {len(pred_features)} features: {pred_features}")

    # Only drop rows missing the target. Impute missing features with training
    # medians so G5 teams (which disproportionately lack portal/recruiting data)
    # stay in the training set — previously this dropna cut training from ~1200
    # to ~300 rows, biased toward P4.
    train = train.dropna(subset=["next_year_rating"])
    train_medians = train[pred_features].median(numeric_only=True)
    train = fill_features(train, pred_features, train_medians)

    X_train = train[pred_features]
    y_train = train["next_year_rating"].values
    train_years = train["year"].values

    # Year-based CV to prevent within-year distributional leakage
    year_cv = YearBasedTimeSeriesSplit(n_splits=4)

    # Pipeline wraps scaler inside CV so validation folds aren't seen by scaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMRegressor(verbose=-1, random_state=42)),
    ])

    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [5, 10, 20],
        "model__subsample": [0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.0, 0.1, 1.0],
    }

    grid_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=60,
        cv=year_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    print("  Running hyperparameter search (this may take a moment)...")
    grid_search.fit(X_train, y_train, groups=train_years)

    best_pipeline = grid_search.best_estimator_
    best_scaler = best_pipeline.named_steps["scaler"]
    best_model = best_pipeline.named_steps["model"]
    # Extract model-only params (strip "model__" prefix)
    best_params = {
        k.replace("model__", ""): v
        for k, v in grid_search.best_params_.items()
        if k.startswith("model__")
    }
    print(f"  Best params: {best_params}")
    print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.3f}")

    # Save feature importances
    importances = dict(zip(pred_features, best_model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(DATA_DIR, "feature_importances.json"), "w") as f:
        json.dump(importances, f, indent=2)
    print(f"  Saved {os.path.join(DATA_DIR, 'feature_importances.json')}")

    # Compute out-of-fold residual std for prediction intervals (year-based)
    oof_residuals = []
    oof_predictions = []
    for train_idx, val_idx in year_cv.split(X_train, y_train, groups=train_years):
        fold_scaler = StandardScaler()
        X_tr_fold = pd.DataFrame(
            fold_scaler.fit_transform(X_train.iloc[train_idx]),
            columns=pred_features,
        )
        X_val_fold = pd.DataFrame(
            fold_scaler.transform(X_train.iloc[val_idx]),
            columns=pred_features,
        )
        fold_model = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=42)
        fold_model.fit(X_tr_fold, y_train[train_idx])
        fold_preds = fold_model.predict(X_val_fold)
        oof_residuals.extend(y_train[val_idx] - fold_preds)
        oof_predictions.extend(fold_preds)

    # Calibrate interval width: use empirical 90th percentile of |residuals|
    # so that 1.645 * residual_std exactly spans the nominal 90% coverage bound
    oof_residuals = np.array(oof_residuals)
    oof_preds_all = np.array(oof_predictions)
    empirical_90th = np.percentile(np.abs(oof_residuals), 90)
    residual_std = empirical_90th / 1.645
    raw_std = np.std(oof_residuals)
    print(f"  Out-of-fold residual std (raw): {raw_std:.3f}")
    print(f"  Calibrated residual std (90th pct / 1.645): {residual_std:.3f}")

    # Heteroskedasticity check: does |residual| correlate with predicted value?
    hetero_r, hetero_p = spearmanr(np.abs(oof_residuals), oof_preds_all)
    print(f"  Heteroskedasticity check: Spearman r={hetero_r:.3f}, p={hetero_p:.4f}", end="")
    quantile_std_map = None
    if hetero_p < 0.05:
        print(" -- HETEROSKEDASTIC -- using quantile-based interval widths")
        # Build per-decile 90th-percentile interval widths
        decile_edges = np.percentile(oof_preds_all, np.linspace(0, 100, 11))
        quantile_std_map = {}
        for i in range(10):
            lo, hi = decile_edges[i], decile_edges[i + 1]
            mask = (oof_preds_all >= lo) & (oof_preds_all <= hi)
            if mask.sum() > 5:
                q90 = np.percentile(np.abs(oof_residuals[mask]), 90)
                quantile_std_map[i] = float(q90 / 1.645)
        print(f"    Decile std range: [{min(quantile_std_map.values()):.2f}, "
              f"{max(quantile_std_map.values()):.2f}]")
    else:
        print(" -- homoskedastic -- constant interval width OK")

    # Naive baseline: persistence (next year = this year)
    baseline_preds = train["rating"].values
    baseline_rmse = np.sqrt(mean_squared_error(y_train, baseline_preds))
    baseline_r2 = r2_score(y_train, baseline_preds)
    print(f"\n  Persistence baseline (next year = this year):")
    print(f"    RMSE: {baseline_rmse:.3f}, R²: {baseline_r2:.3f}")

    return best_model, best_scaler, residual_std, pred_features, best_params, quantile_std_map


def predict_multi_year(model, scaler, df_flat, roster_agg, pred_features, residual_std,
                       decay_rate=None, quantile_std_map=None,
                       ranker=None, ranker_scaler=None, conformal_half_width=None):
    """Generate chained multi-year predictions with ensemble decay and prediction intervals.

    If quantile_std_map is provided (heteroskedastic case), interval widths vary by
    predicted-value decile. Otherwise constant calibrated residual_std is used.

    Returns: predictions DataFrame, validation results DataFrame
    """
    df = df_flat.copy()

    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")

    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)

    # Per-feature medians from the full historical sample — used to impute
    # missing features so Group-of-5 / Independent teams aren't excluded from
    # predictions just because they lack portal or position-recruiting coverage.
    feature_medians = df[pred_features].median(numeric_only=True)

    def _impute(frame):
        return frame.assign(**{
            c: frame[c].fillna(feature_medians[c]) for c in pred_features
        })

    # --- Validation: predict 2025 from 2024 features ---
    val_input = _impute(df[df["year"] == 2024].copy())
    val_input = val_input.dropna(subset=["rating"])  # need a team to exist
    X_val = pd.DataFrame(
        scaler.transform(val_input[pred_features]),
        columns=pred_features, index=val_input.index,
    )
    val_preds = model.predict(X_val)

    val_results = val_input[["team", "conference"]].copy()
    val_results["predicted_rating"] = val_preds
    actual_2025 = df[df["year"] == 2025][["team", "rating"]].rename(
        columns={"rating": "actual_rating"}
    )
    val_results = val_results.merge(actual_2025, on="team", how="inner")
    corr = val_results["predicted_rating"].corr(val_results["actual_rating"])
    print(f"  2025 validation correlation: {corr:.3f} (n={len(val_results)})")

    # --- Multi-year chained predictions ---
    if decay_rate is None:
        decay_rate = config.decay_rate
    bootstrap_n = config.bootstrap_n
    confidence_level = config.confidence_level
    alpha = 1 - confidence_level

    base = _impute(df[df["year"] == 2025].copy())
    base = base.dropna(subset=["rating"])
    league_mean = df[df["year"] == 2025]["rating"].mean()
    print(f"  Projecting {len(base)} teams forward "
          f"({base['conference'].nunique()} conferences)")

    all_predictions = []
    current_base = base.copy()

    for k, pred_year in enumerate(config.prediction_years, start=1):
        X = pd.DataFrame(
            scaler.transform(current_base[pred_features]),
            columns=pred_features, index=current_base.index,
        )
        raw_preds = model.predict(X)

        # Ensemble decay toward league mean
        decay_factor = decay_rate ** k
        decayed_preds = decay_factor * raw_preds + (1 - decay_factor) * league_mean

        # Prediction intervals — conformal if available (empirical residual quantile),
        # else fall back to the original Gaussian bootstrap. sqrt(k) scales horizon.
        if conformal_half_width is not None:
            half_width = conformal_half_width * np.sqrt(k)
            lower = decayed_preds - half_width
            upper = decayed_preds + half_width
        else:
            rng = np.random.RandomState(42 + k)
            if quantile_std_map:
                pred_deciles = pd.qcut(
                    pd.Series(decayed_preds), q=10, labels=False, duplicates="drop"
                ).fillna(4).astype(int).values
                per_team_std = np.array([
                    quantile_std_map.get(int(d), residual_std) * np.sqrt(k)
                    for d in pred_deciles
                ])
                bootstrap_preds = np.array([
                    decayed_preds + rng.normal(0, per_team_std)
                    for _ in range(bootstrap_n)
                ])
            else:
                horizon_std = residual_std * np.sqrt(k)
                bootstrap_preds = np.array([
                    decayed_preds + rng.normal(0, horizon_std, size=len(decayed_preds))
                    for _ in range(bootstrap_n)
                ])
            lower = np.percentile(bootstrap_preds, alpha / 2 * 100, axis=0)
            upper = np.percentile(bootstrap_preds, (1 - alpha / 2) * 100, axis=0)

        year_result = current_base[["team", "conference"]].copy()
        year_result["year"] = pred_year
        year_result["predicted_rating"] = decayed_preds.round(2)
        year_result["pred_lower"] = lower.round(2)
        year_result["pred_upper"] = upper.round(2)

        # Ranking: use the LGBMRanker's ordinal scores when available
        # (Spearman r ~0.87 vs regression ~0.75 on 2025 val)
        if ranker is not None and ranker_scaler is not None:
            ranker_scores = score_with_ranker(
                ranker, ranker_scaler, current_base, pred_features, feature_medians
            )
            year_result["ranker_score"] = ranker_scores
            rank_series = pd.Series(ranker_scores, index=year_result.index)
        else:
            rank_series = year_result["predicted_rating"]

        year_result["national_rank"] = (
            rank_series.rank(ascending=False, method="min").astype("Int64")
        )
        year_result["conference_rank"] = (
            rank_series.groupby(year_result["conference"])
            .rank(ascending=False, method="min").astype("Int64")
        )

        all_predictions.append(year_result)

        # Carry forward for next year's prediction
        next_base = current_base.copy()
        next_base["prior_year_rating"] = next_base["rating"].values
        next_base["rating"] = decayed_preds
        # Talent regresses 5% toward league mean per year (recruiting class turnover)
        if "talent" in next_base.columns:
            league_talent_mean = df[df["year"] == 2025]["talent"].mean()
            next_base["talent"] = next_base["talent"] * 0.95 + league_talent_mean * 0.05
        current_base = next_base

    predictions = pd.concat(all_predictions, ignore_index=True)
    return predictions, val_results


def compute_nil_analysis(roster_agg):
    """Analyze talent concentration pre/post NIL (2022+).

    Outputs:
        data/nil_team_analysis.csv  — per-team elite count change
        data/nil_concentration.csv  — yearly Gini coefficient of elite counts
    """
    if roster_agg.empty:
        print("  No roster data — skipping NIL analysis.")
        return

    nil_year = config.nil_start_year

    # Elite count = 4+5 star players per team-year
    ra = roster_agg.copy()
    ra["elite_count"] = ra["star_5_count"] + ra["star_4_count"]

    # --- Per-team analysis ---
    pre = ra[ra["year"] < nil_year].groupby("team")["elite_count"].mean()
    post = ra[ra["year"] >= nil_year].groupby("team")["elite_count"].mean()
    team_df = pd.DataFrame({"pre_nil_avg_elite": pre, "post_nil_avg_elite": post})
    team_df["elite_delta"] = team_df["post_nil_avg_elite"] - team_df["pre_nil_avg_elite"]
    team_df = team_df.dropna().sort_values("elite_delta", ascending=False).reset_index()
    team_df.to_csv(os.path.join(DATA_DIR, "nil_team_analysis.csv"), index=False)
    print(f"  Saved data/nil_team_analysis.csv ({len(team_df)} teams)")

    # --- Year-by-year Gini coefficient ---
    def gini(values):
        v = np.sort(np.array(values, dtype=float))
        n = len(v)
        if n == 0 or v.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))

    gini_records = []
    for year, grp in ra.groupby("year"):
        gini_records.append({
            "year": year,
            "gini_elite": round(gini(grp["elite_count"].values), 4),
            "total_elite": int(grp["elite_count"].sum()),
            "num_teams": len(grp),
        })
    gini_df = pd.DataFrame(gini_records).sort_values("year")
    gini_df.to_csv(os.path.join(DATA_DIR, "nil_concentration.csv"), index=False)
    print(f"  Saved data/nil_concentration.csv ({len(gini_df)} years)")


def compute_evaluation_metrics(y_true, y_pred, y_lower=None, y_upper=None):
    """Compute evaluation metrics: R², RMSE, MAE, Spearman r, interval coverage."""
    sp_r, sp_p = spearmanr(y_true, y_pred)
    metrics = {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "correlation": round(float(np.corrcoef(y_true, y_pred)[0, 1]), 4),
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
    }

    if y_lower is not None and y_upper is not None:
        coverage = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
        metrics["interval_coverage"] = round(coverage, 4)

    return metrics


def load_bt_features(games_path=None):
    if games_path is None:
        games_path = os.path.join(DATA_DIR, "game_results.csv")
    """Compute Bradley-Terry strength per team per year from game outcomes.

    Uses choix.ilsr (Iterative Luce Spectral Ranking) with L2 regularisation
    (alpha>0) to handle near-disconnected graph components.

    Returns a DataFrame with columns: year, team, bt_strength.
    Returns empty DataFrame if choix is not installed or data is missing.
    """
    try:
        import choix
    except ImportError:
        print("  choix not installed — skipping BT features (pip install choix)")
        return pd.DataFrame()

    if not Path(games_path).exists():
        print(f"  {games_path} not found — skipping BT features")
        return pd.DataFrame()

    games = pd.read_csv(games_path)
    games = games.dropna(subset=["home_points", "away_points"])

    bt_records = []
    for year, grp in games.groupby("year"):
        teams = sorted(set(grp["home_team"].tolist() + grp["away_team"].tolist()))
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        pairs = []
        for _, row in grp.iterrows():
            h, a = row["home_team"], row["away_team"]
            hp, ap = int(row["home_points"]), int(row["away_points"])
            if hp == ap:
                continue
            winner, loser = (h, a) if hp > ap else (a, h)
            if winner in team_idx and loser in team_idx:
                pairs.append((team_idx[winner], team_idx[loser]))

        if len(pairs) < n:
            print(f"  BT {year}: only {len(pairs)} pairs for {n} teams — skipping")
            continue

        try:
            # ilsr_pairwise: L2 regularisation (alpha>0) handles sparse/near-disconnected graphs
            params = choix.ilsr_pairwise(n, pairs, alpha=0.01)
            for team, idx in team_idx.items():
                bt_records.append({"year": year, "team": team, "bt_strength": float(params[idx])})
        except Exception as e:
            print(f"  BT {year}: failed ({e}) — skipping")

    if not bt_records:
        return pd.DataFrame()

    bt_df = pd.DataFrame(bt_records)
    print(f"  BT features: {len(bt_df)} team-years across {bt_df['year'].nunique()} seasons")
    return bt_df


def load_coach_features(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "coaches.csv")
    """Load coaching continuity features (tenure, first-year flag)."""
    if not Path(path).exists():
        print(f"  {path} not found — skipping coach features")
        return pd.DataFrame()
    df = pd.read_csv(path)[["year", "team", "coach_tenure", "is_first_year_coach"]]
    print(f"  Coach features: {len(df)} team-years")
    return df


def load_portal_features(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "transfer_portal.csv")
    """Load transfer portal net-flow features."""
    if not Path(path).exists():
        print(f"  {path} not found — skipping portal features")
        return pd.DataFrame()
    cols = ["year", "team", "portal_net_count", "portal_net_rating", "portal_inbound_count"]
    available = [c for c in cols if c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path)[available]
    print(f"  Portal features: {len(df)} team-years")
    return df


def merge_auxiliary_features(df_flat, bt_df, coach_df, portal_df, pos_rec_df=None):
    """Merge BT, coaching, and portal features into the main flat DataFrame."""
    df = df_flat.copy()
    aux_list = [(bt_df, "BT"), (coach_df, "coach"), (portal_df, "portal")]
    if pos_rec_df is not None and not pos_rec_df.empty:
        aux_list.append((pos_rec_df, "position recruiting"))
    for aux, name in aux_list:
        if not aux.empty:
            df = df.merge(aux, on=["year", "team"], how="left")
            feat_cols = [c for c in aux.columns if c not in ("year", "team")]
            nan_rate = df[feat_cols].isna().mean().mean()
            print(f"  Merged {name} features — NaN rate: {nan_rate:.1%}")

    # BT: fill missing current-year strength with 0 (neutral), then create lagged feature.
    # We use prior_bt_strength (year T-1) rather than current bt_strength (year T)
    # to avoid multicollinearity with SP+ rating which also reflects year-T performance.
    if "bt_strength" in df.columns:
        df["bt_strength"] = df["bt_strength"].fillna(0.0)
        df = df.sort_values(["team", "year"])
        df["prior_bt_strength"] = df.groupby("team")["bt_strength"].shift(1).fillna(0.0)
        # Drop raw bt_strength — we only expose the lagged version as a feature
        df = df.drop(columns=["bt_strength"])
        print(f"  Created prior_bt_strength (lagged BT, avoids SP+ multicollinearity)")
    return df


def compute_baseline_comparison(df_flat, roster_agg, pred_features, val_years=None):
    """Compare rolling OOS RMSE: LightGBM vs OLS Ridge vs Exponential Smoothing.

    Reports RMSE for each baseline model across the same validation years
    as compute_rolling_validation. Results printed and returned as dict.
    """
    from sklearn.linear_model import Ridge

    if val_years is None:
        val_years = list(range(2020, 2026))

    df = df_flat.copy()
    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")
    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)

    ridge_true, ridge_pred = [], []
    ets_true, ets_pred = [], []

    for Y in val_years:
        train_df = df[(df["year"] >= 2016) & (df["year"] <= Y - 2)].copy()
        train_df["next_year_rating"] = train_df.groupby("team")["rating"].shift(-1)
        train_df = train_df.dropna(subset=pred_features + ["next_year_rating"])
        if len(train_df) < 10:
            continue

        # OLS Ridge
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(train_df[pred_features])
        y_tr = train_df["next_year_rating"].values
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_tr)

        val_df = df[df["year"] == Y - 1].dropna(subset=pred_features).copy()
        actual_df = df[df["year"] == Y][["team", "rating"]].rename(columns={"rating": "actual"})
        if val_df.empty:
            continue

        preds_ridge = ridge.predict(fold_scaler.transform(val_df[pred_features]))
        vd = val_df[["team"]].copy()
        vd["predicted"] = preds_ridge
        vd = vd.merge(actual_df, on="team", how="inner")
        if not vd.empty:
            ridge_true.extend(vd["actual"].tolist())
            ridge_pred.extend(vd["predicted"].tolist())

        # Exponential smoothing: 0.7 * current + 0.3 * prior (simple ETS approx)
        ets_base = df[df["year"] == Y - 1][["team", "rating", "prior_year_rating"]].copy()
        ets_base["ets_pred"] = (
            0.7 * ets_base["rating"].fillna(ets_base["prior_year_rating"])
            + 0.3 * ets_base["prior_year_rating"].fillna(ets_base["rating"])
        )
        ets_merged = ets_base.merge(actual_df, on="team", how="inner").dropna(subset=["ets_pred"])
        if not ets_merged.empty:
            ets_true.extend(ets_merged["actual"].tolist())
            ets_pred.extend(ets_merged["ets_pred"].tolist())

    results = {}
    if ridge_true:
        results["ridge_rmse"] = round(float(np.sqrt(mean_squared_error(ridge_true, ridge_pred))), 4)
    if ets_true:
        results["ets_rmse"] = round(float(np.sqrt(mean_squared_error(ets_true, ets_pred))), 4)

    print(f"\n  Baseline comparison (rolling 2020-2025):")
    for k, v in results.items():
        print(f"    {k}: {v:.3f}")
    return results


def load_position_recruiting(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "position_recruiting.csv")
    """Load per-position-group continuous recruiting quality features."""
    if not Path(path).exists():
        print(f"  {path} not found — skipping position recruiting features")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Position recruiting features: {len(df)} team-years, "
          f"{[c for c in df.columns if c.startswith('rec_')]}")
    return df


def train_lgbm_ranker(df_flat, roster_agg, pred_features, best_params):
    """Train a LightGBMRanker to predict SP+ rank (ordinal target).

    Uses lambdarank objective with quantile-based relevance labels.
    Computes Spearman r on 2025 validation set and compares to regression model.

    Returns: ranker model, scaler, spearman_r on validation.
    """
    df = df_flat.copy()
    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")
    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)
    df["next_year_rating"] = df.groupby("team")["rating"].shift(-1)

    train = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()
    # Keep all FBS teams — only drop if missing target. Impute features.
    train = train.dropna(subset=["next_year_rating"])
    train_medians = train[pred_features].median(numeric_only=True)
    train = fill_features(train, pred_features, train_medians)

    # Relevance labels: decile of next_year_rating within each year (0=worst, 9=best)
    train["rank_label"] = train.groupby("year")["next_year_rating"].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
    ).fillna(4).astype(int)

    # LGBMRanker requires sorted-by-group data and group sizes
    train = train.sort_values("year")
    group_sizes = train.groupby("year").size().values

    ranker_scaler = StandardScaler()
    X_tr = ranker_scaler.fit_transform(train[pred_features])
    y_tr = train["rank_label"].values

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=best_params.get("n_estimators", 300),
        max_depth=best_params.get("max_depth", 3),
        learning_rate=best_params.get("learning_rate", 0.05),
        num_leaves=best_params.get("num_leaves", 15),
        verbose=-1,
        random_state=42,
    )
    ranker.fit(X_tr, y_tr, group=group_sizes)

    # Validate on 2025: Spearman r between ranker scores and actual SP+
    val_df = df[df["year"] == 2024].copy()
    val_df = fill_features(val_df, pred_features, train_medians)
    actual_2025 = df[df["year"] == 2025][["team", "rating"]].rename(
        columns={"rating": "actual_rating"}
    )
    if val_df.empty or actual_2025.empty:
        return ranker, ranker_scaler, None

    X_val = ranker_scaler.transform(val_df[pred_features])
    ranker_scores = ranker.predict(X_val)
    val_df = val_df[["team"]].copy()
    val_df["ranker_score"] = ranker_scores
    val_df = val_df.merge(actual_2025, on="team", how="inner")

    ranker_sp, _ = spearmanr(val_df["ranker_score"], val_df["actual_rating"])
    print(f"  LGBMRanker 2025 Spearman r: {ranker_sp:.4f}")
    return ranker, ranker_scaler, float(ranker_sp)


def score_with_ranker(ranker, ranker_scaler, frame, pred_features, feature_medians):
    """Apply a trained LGBMRanker to a frame of team-years, imputing missing
    features with the provided medians. Returns a 1-D numpy array of scores."""
    filled = frame[pred_features].copy()
    for c in pred_features:
        filled[c] = filled[c].fillna(feature_medians[c])
    X = ranker_scaler.transform(filled)
    return ranker.predict(X)


def fill_features(frame, pred_features, medians):
    """Median-impute the specified features in-place on a copy of frame."""
    out = frame.copy()
    for c in pred_features:
        if c in out.columns:
            out[c] = out[c].fillna(medians[c])
    return out


def compute_rolling_validation(df_flat, roster_agg, best_params, pred_features, val_years=None):
    """Walk-forward validation across multiple held-out years.

    For each year Y in val_years, trains on years [2016, Y-2] and predicts Y
    from year Y-1 features. Avoids any data leakage from the validation year.

    Returns dict with per-year RMSE and overall rolling RMSE.
    """
    if val_years is None:
        val_years = list(range(2020, 2026))

    df = df_flat.copy()
    if not roster_agg.empty:
        df = df.merge(roster_agg, on=["year", "team"], how="left")

    df = df.sort_values(["team", "year"])
    df["prior_year_rating"] = df.groupby("team")["rating"].shift(1)

    per_year_rmse = {}
    all_y_true = []
    all_y_pred = []
    all_conferences = []

    for Y in val_years:
        # Train on rows where year in [2016, Y-2] with target = next_year_rating
        train_df = df[(df["year"] >= 2016) & (df["year"] <= Y - 2)].copy()
        train_df["next_year_rating"] = train_df.groupby("team")["rating"].shift(-1)
        train_df = train_df.dropna(subset=["next_year_rating"])
        fold_medians = train_df[pred_features].median(numeric_only=True)
        train_df = fill_features(train_df, pred_features, fold_medians)

        if len(train_df) < 10:
            print(f"  Rolling val {Y}: skipped (insufficient training data)")
            continue

        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(train_df[pred_features])
        y_tr = train_df["next_year_rating"].values

        fold_model = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=42)
        fold_model.fit(X_tr, y_tr)

        # Predict from year Y-1 features → year Y (impute so G5 teams stay in)
        val_df = df[df["year"] == Y - 1].copy()
        val_df = fill_features(val_df, pred_features, fold_medians)
        actual_df = df[df["year"] == Y][["team", "rating", "conference"]].rename(
            columns={"rating": "actual"}
        )

        if val_df.empty:
            continue

        preds = fold_model.predict(fold_scaler.transform(val_df[pred_features]))
        val_df = val_df[["team"]].copy()
        val_df["predicted"] = preds
        val_df = val_df.merge(actual_df, on="team", how="inner")

        if val_df.empty:
            continue

        rmse = float(np.sqrt(mean_squared_error(val_df["actual"], val_df["predicted"])))
        per_year_rmse[Y] = round(rmse, 4)
        all_y_true.extend(val_df["actual"].tolist())
        all_y_pred.extend(val_df["predicted"].tolist())
        all_conferences.extend(val_df["conference"].tolist())
        print(f"  Rolling val {Y}: RMSE={rmse:.3f} ({len(val_df)} teams)")

    overall = float(np.sqrt(mean_squared_error(all_y_true, all_y_pred))) if all_y_true else None

    # Conformal calibration: empirical 90th percentile of |residuals|
    # becomes the half-width for 90% intervals at horizon k=1.
    residuals = np.array(all_y_true) - np.array(all_y_pred) if all_y_true else np.array([])
    conformal_half_width = (
        float(np.percentile(np.abs(residuals), 90)) if len(residuals) else None
    )

    # Slice-based RMSE by conference (Huyen ch.6)
    by_conference = {}
    if all_y_true:
        slice_df = pd.DataFrame({
            "actual": all_y_true, "predicted": all_y_pred, "conference": all_conferences,
        })
        for conf, grp in slice_df.groupby("conference"):
            if len(grp) >= 5:
                by_conference[conf] = {
                    "rmse": round(float(np.sqrt(mean_squared_error(grp["actual"], grp["predicted"]))), 4),
                    "n": int(len(grp)),
                }

    return {
        "per_year": per_year_rmse,
        "overall_rmse": round(overall, 4) if overall is not None else None,
        "conformal_half_width": round(conformal_half_width, 4) if conformal_half_width else None,
        "by_conference": by_conference,
    }


def main():
    # 1. Load data
    print("Loading advanced metrics...")
    df_flat = load_advanced_metrics()

    print("Loading roster aggregates...")
    roster_agg = load_roster_aggregates()
    if roster_agg.empty:
        print("  No merged roster data found — proceeding without recruiting features.")
    else:
        print(f"  Loaded roster aggregates for {len(roster_agg)} team-years.")

    # 2. Load auxiliary features (BT, coaches, portal, position recruiting)
    print("\nLoading auxiliary features...")
    bt_df = load_bt_features()
    coach_df = load_coach_features()
    portal_df = load_portal_features()
    pos_rec_df = load_position_recruiting()
    df_flat = merge_auxiliary_features(df_flat, bt_df, coach_df, portal_df, pos_rec_df)

    # 3. Estimate empirical decay rate from historical AR(1)
    print("\nEstimating empirical decay rate...")
    estimated_decay = estimate_decay_rate(df_flat)

    # 4. Train LightGBM model (full training set) → best_params, feature importances
    print("\nTraining LightGBM model...")
    best_model, scaler, residual_std, pred_features, best_params, quantile_std_map = train_lgbm_model(
        df_flat, roster_agg
    )

    # 4b. Train LightGBM Ranker (ordinal model) — compare Spearman r to regression
    print("\nTraining LightGBM Ranker (ordinal model)...")
    ranker, ranker_scaler, ranker_spearman = train_lgbm_ranker(
        df_flat, roster_agg, pred_features, best_params
    )

    # 4c. Rolling out-of-sample validation — also calibrates conformal intervals
    print("\nComputing rolling out-of-sample validation...")
    rolling_val = compute_rolling_validation(df_flat, roster_agg, best_params, pred_features)
    conformal_q = rolling_val.get("conformal_half_width")
    if conformal_q:
        print(f"  Conformal 90% half-width (rolling val): {conformal_q:.3f}")

    # 4d. Pick whichever model (regression or ranker) has better 2024→2025 Spearman
    # for producing the final rank ordering. Median-impute to match training.
    _dv = df_flat.merge(roster_agg, on=["year", "team"], how="left") if not roster_agg.empty else df_flat.copy()
    _dv = _dv.sort_values(["team", "year"])
    _dv["prior_year_rating"] = _dv.groupby("team")["rating"].shift(1)
    _medians = _dv[pred_features].median(numeric_only=True)
    _val24 = fill_features(_dv[_dv["year"] == 2024], pred_features, _medians)
    _act25 = _dv[_dv["year"] == 2025][["team", "rating"]]
    _reg_preds = best_model.predict(scaler.transform(_val24[pred_features]))
    _merged = _val24[["team"]].assign(pred=_reg_preds).merge(_act25, on="team", how="inner")
    reg_spearman = float(spearmanr(_merged["pred"], _merged["rating"])[0])
    print(f"  Model selection: regression Spearman={reg_spearman:.4f}, "
          f"ranker Spearman={ranker_spearman:.4f}")
    use_ranker = ranker_spearman is not None and ranker_spearman > reg_spearman
    print(f"  -> Using {'RANKER' if use_ranker else 'REGRESSION'} for rank ordering")

    # 5. Walk-forward historical rankings
    print("\nComputing walk-forward power rankings...")
    df = compute_rankings(
        df_flat, roster_agg, best_params,
        ranker=ranker if use_ranker else None,
        ranker_scaler=ranker_scaler if use_ranker else None,
        pred_features_override=pred_features,
    )

    output_cols_national = ["year", "team", "conference", "predicted_rating", "national_rank"]
    output_cols_conference = ["year", "team", "conference", "predicted_rating", "conference_rank"]

    national = df[output_cols_national].sort_values(["year", "national_rank"])
    conference = df[output_cols_conference].sort_values(["year", "conference", "conference_rank"])

    national.to_csv(os.path.join(DATA_DIR, "power_rankings_national.csv"), index=False)
    conference.to_csv(os.path.join(DATA_DIR, "power_rankings_conference.csv"), index=False)

    print(f"Saved data/power_rankings_national.csv ({len(national)} rows)")
    print(f"Saved data/power_rankings_conference.csv ({len(conference)} rows)")

    latest = df["year"].max()
    top10 = national[national["year"] == latest].head(10)
    print(f"\n=== {latest} National Top 10 ===")
    print(top10.to_string(index=False))

    # 5. Multi-year predictions (2026-2028) — conformal intervals, ranker-based ranks
    print(f"\nGenerating {config.prediction_years} predictions...")
    predictions, val_results = predict_multi_year(
        best_model, scaler, df_flat, roster_agg, pred_features, residual_std,
        decay_rate=estimated_decay, quantile_std_map=quantile_std_map,
        ranker=ranker if use_ranker else None,
        ranker_scaler=ranker_scaler if use_ranker else None,
        conformal_half_width=conformal_q,
    )

    pred_output = predictions[
        ["year", "team", "conference", "predicted_rating", "pred_lower", "pred_upper",
         "national_rank", "conference_rank"]
    ].sort_values(["year", "national_rank"])
    pred_output.to_csv(os.path.join(DATA_DIR, "power_rankings_predictions.csv"), index=False)
    print(f"Saved data/power_rankings_predictions.csv ({len(pred_output)} rows)")

    # 6. Evaluation metrics
    val_results.to_csv(os.path.join(DATA_DIR, "validation_results.csv"), index=False)
    print(f"Saved data/validation_results.csv ({len(val_results)} rows)")

    val_metrics = compute_evaluation_metrics(
        val_results["actual_rating"].values,
        val_results["predicted_rating"].values,
    )
    # Report coverage under both the Gaussian (legacy) and conformal (new) widths
    half_width = conformal_q if conformal_q else 1.645 * residual_std
    val_lower = val_results["predicted_rating"].values - half_width
    val_upper = val_results["predicted_rating"].values + half_width
    interval_metrics = compute_evaluation_metrics(
        val_results["actual_rating"].values,
        val_results["predicted_rating"].values,
        val_lower,
        val_upper,
    )
    val_metrics["interval_coverage"] = interval_metrics.get("interval_coverage", None)

    # Persistence baseline on validation set
    baseline_val_preds = val_results.merge(
        df_flat[df_flat["year"] == 2024][["team", "rating"]].rename(
            columns={"rating": "baseline_pred"}
        ),
        on="team", how="inner",
    )
    if len(baseline_val_preds) > 0:
        bl_rmse = np.sqrt(mean_squared_error(
            baseline_val_preds["actual_rating"], baseline_val_preds["baseline_pred"]
        ))
        bl_r2 = r2_score(
            baseline_val_preds["actual_rating"], baseline_val_preds["baseline_pred"]
        )
        val_metrics["baseline_rmse"] = round(float(bl_rmse), 4)
        val_metrics["baseline_r2"] = round(float(bl_r2), 4)
        print(f"\n  Persistence baseline (2025 validation): RMSE={bl_rmse:.3f}, R²={bl_r2:.3f}")

    # Rolling out-of-sample validation already computed above — reuse results
    val_metrics["rolling_val_rmse"] = rolling_val["overall_rmse"]
    val_metrics["rolling_val_per_year"] = rolling_val["per_year"]
    val_metrics["estimated_decay_rate"] = round(estimated_decay, 4)
    if rolling_val.get("conformal_half_width"):
        val_metrics["conformal_half_width"] = round(rolling_val["conformal_half_width"], 4)

    # Slice-based evaluation by conference (Huyen, Designing ML Systems ch.6)
    slice_metrics = {"by_conference_rolling_val": rolling_val.get("by_conference", {})}
    if not val_results.empty and "conference" in val_results.columns:
        by_conf_2025 = {}
        for conf, grp in val_results.groupby("conference"):
            if len(grp) >= 3:
                sp_r, _ = spearmanr(grp["predicted_rating"], grp["actual_rating"])
                by_conf_2025[conf] = {
                    "rmse": round(float(np.sqrt(mean_squared_error(
                        grp["actual_rating"], grp["predicted_rating"]
                    ))), 4),
                    "spearman_r": round(float(sp_r), 4) if not np.isnan(sp_r) else None,
                    "n": int(len(grp)),
                }
        slice_metrics["by_conference_2025_val"] = by_conf_2025
    val_metrics["slice_metrics"] = slice_metrics
    if ranker_spearman is not None:
        val_metrics["ranker_spearman_r"] = round(ranker_spearman, 4)
        reg_sp = val_metrics.get("spearman_r", None)
        if reg_sp:
            print(f"\n  Regression Spearman r: {reg_sp:.4f} | Ranker Spearman r: {ranker_spearman:.4f}")

    # Baseline comparison: OLS Ridge + Exponential Smoothing
    print("\nComputing baseline model comparison...")
    baselines = compute_baseline_comparison(df_flat, roster_agg, pred_features)
    val_metrics.update(baselines)

    with open(os.path.join(DATA_DIR, "model_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\nModel metrics: {val_metrics}")

    # 7. NIL analysis
    print("\nRunning NIL talent concentration analysis...")
    compute_nil_analysis(roster_agg)

    # Print top 10 for each prediction year
    for year in config.prediction_years:
        year_preds = pred_output[pred_output["year"] == year].head(10)
        print(f"\n=== Projected {year} Top 10 ===")
        print(year_preds.to_string(index=False))


if __name__ == "__main__":
    main()
