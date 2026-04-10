"""
Engineer matchup features from starters and game results for win probability model.

Reads starters_by_season.csv and game_results.csv, computes positional
advantage features for each game, and splits into training data and
upcoming-season data for the dashboard.

Outputs:
  data/Xy_train.csv         — training features + target
  data/2025_games_clean.csv — upcoming season games (for dashboard)
  data/2025_rosters.csv     — upcoming season rosters with aggregates (for dashboard)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import pandas as pd
import config

FEATURES = [
    "QB_diff", "OL_adv_1", "OL_adv_2", "WR_adv_1", "WR_adv_2",
    "RB_adv_1", "RB_adv_2", "TE_diff", "PK_diff", "P_diff",
    "QB1_vs_DB_2", "QB2_vs_DB_1", "PASS_SYNERGY_diff", "team_1_is_home",
]


def add_group_averages(teams):
    """Compute positional group averages from individual starter ratings."""
    teams["OL_avg"] = teams[[f"OL{i}" for i in range(1, 6)]].mean(axis=1)
    teams["WR_avg"] = teams[[f"WR{i}" for i in range(1, 3)]].mean(axis=1)
    teams["DL_avg"] = teams[[f"DL{i}" for i in range(1, 5)]].mean(axis=1)
    teams["LB_avg"] = teams[[f"LB{i}" for i in range(1, 4)]].mean(axis=1)
    teams["DB_avg"] = teams[[f"DB{i}" for i in range(1, 5)]].mean(axis=1)
    return teams


def build_matchup_row(year, team_1, team_2, team_1_is_home, team_1_won, teams):
    t1 = teams[(teams.season == year) & (teams.team == team_1)]
    t2 = teams[(teams.season == year) & (teams.team == team_2)]

    team_1_synergy = (
        2.0 * t1.QB1.iloc[0] + 1.0 * t1.WR_avg.iloc[0] + 0.8 * t1.TE1.iloc[0]
    ) / 3.8
    team_2_synergy = (
        2.0 * t2.QB1.iloc[0] + 1.0 * t2.WR_avg.iloc[0] + 0.8 * t2.TE1.iloc[0]
    ) / 3.8

    return {
        "QB_diff": t1.QB1.iloc[0] - t2.QB1.iloc[0],
        "OL_adv_1": t1.OL_avg.iloc[0] - t2.DL_avg.iloc[0],
        "OL_adv_2": t2.OL_avg.iloc[0] - t1.DL_avg.iloc[0],
        "WR_adv_1": t1.WR_avg.iloc[0] - t2.DB_avg.iloc[0],
        "WR_adv_2": t2.WR_avg.iloc[0] - t1.DB_avg.iloc[0],
        "RB_adv_1": t1.RB1.iloc[0] - t2.LB_avg.iloc[0],
        "RB_adv_2": t2.RB1.iloc[0] - t1.LB_avg.iloc[0],
        "TE_diff": t1.TE1.iloc[0] - t2.TE1.iloc[0],
        "PK_diff": t1.PK1.iloc[0] - t2.PK1.iloc[0],
        "P_diff": t1.P1.iloc[0] - t2.P1.iloc[0],
        "QB1_vs_DB_2": t1.QB1.iloc[0] - t2.DB_avg.iloc[0],
        "QB2_vs_DB_1": t2.QB1.iloc[0] - t1.DB_avg.iloc[0],
        "PASS_SYNERGY_diff": team_1_synergy - team_2_synergy,
        "team_1_is_home": team_1_is_home,
        "season": year,
        "team_1": team_1,
        "team_2": team_2,
        "target": team_1_won,
    }


def build_xy(df, teams, train=True):
    rows = []
    for year, home, away, home_won in df[
        ["season", "home_team", "away_team", "target"]
    ].itertuples(index=False):
        rows.append(build_matchup_row(year, home, away, 1, home_won, teams))
        if train:
            rows.append(build_matchup_row(year, away, home, 0, 1 - home_won, teams))
    return pd.DataFrame(rows)


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")

    teams = pd.read_csv(os.path.join(data_dir, "starters_by_season.csv"))
    games = pd.read_csv(os.path.join(data_dir, "game_results.csv"))

    # game_results.csv uses snake_case; remap to match the feature builder
    games = games.rename(columns={
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "home_points": "homePoints",
        "away_points": "awayPoints",
        "year": "season",
        "season_type": "seasonType",
        "start_date": "startDate",
    })

    teams = teams.fillna(0.0)

    # Remove games where either team has no roster data
    missing_teams = set()
    for _, game in games.iterrows():
        season = game.season
        for team_col in ("homeTeam", "awayTeam"):
            team = game[team_col]
            if len(teams[(teams["season"] == season) & (teams["team"] == team)]) < 1:
                missing_teams.add((season, team))

    games["home_key"] = list(zip(games["season"], games["homeTeam"]))
    games["away_key"] = list(zip(games["season"], games["awayTeam"]))
    games_clean = games[
        ~games["home_key"].isin(missing_teams)
        & ~games["away_key"].isin(missing_teams)
    ]
    games_clean = games_clean[games_clean.homePoints != games_clean.awayPoints]

    # Save upcoming season games for dashboard
    upcoming = config.upcoming_season
    upcoming_games = games_clean[
        (games_clean.season == upcoming) & (games_clean.seasonType == "regular")
    ]
    upcoming_games.to_csv(
        os.path.join(data_dir, "2025_games_clean.csv"), index=False
    )
    print(f"Saved 2025_games_clean.csv ({len(upcoming_games)} games)")

    # Add target and group averages
    games_clean = games_clean.copy()
    games_clean["target"] = (games_clean["homePoints"] > games_clean["awayPoints"]).astype(int)
    teams = add_group_averages(teams)

    # Save upcoming rosters for dashboard
    upcoming_rosters = teams[teams.season == upcoming]
    upcoming_rosters.to_csv(
        os.path.join(data_dir, "2025_rosters.csv"), index=False
    )
    print(f"Saved 2025_rosters.csv ({len(upcoming_rosters)} rosters)")

    # Remap column names back to snake_case for build_xy
    games_clean = games_clean.rename(columns={
        "homeTeam": "home_team",
        "awayTeam": "away_team",
    })

    # Build training set
    train_years = list(range(config.train_start, upcoming))
    train_data = games_clean[games_clean.season.isin(train_years)]
    Xy_train = build_xy(train_data, teams)
    Xy_train.to_csv(os.path.join(data_dir, "Xy_train.csv"), index=False)
    print(f"Saved Xy_train.csv ({len(Xy_train)} rows)")


if __name__ == "__main__":
    main()
