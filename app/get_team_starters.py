"""
Identify top-rated starters per team/position from merged roster data.

For each team-season, selects the highest-rated players at each standardized
position group (e.g., top 5 OL, top 4 DB). Players without star ratings
default to 0.0.

Output: data/starters_by_season.csv
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import pandas as pd
import config

POSITION_MAPPING = config.position_mapping
POSITIONS = config.positions


def main():
    team_starters = []

    for season in range(config.season_start, config.season_end + 1):
        csv_path = os.path.join(PROJECT_ROOT, f"data/merged_rosters/{season}_rosters.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {season}: {csv_path} not found")
            continue

        data = pd.read_csv(csv_path)
        data["position"] = data["position"].replace(POSITION_MAPPING)
        teams = data["team"].unique()

        for team in teams:
            roster = data[data["team"] == team]
            starters = {"season": season, "team": team}

            for position, num_starters in POSITIONS.items():
                players_at_pos = roster[roster["position"] == position].sort_values(
                    by=["stars"], ascending=False
                )
                for i in range(num_starters):
                    try:
                        starters[f"{position}{i + 1}"] = players_at_pos["stars"].iloc[i]
                    except (IndexError, KeyError):
                        starters[f"{position}{i + 1}"] = 0.0

            team_starters.append(starters)

    starters_df = pd.DataFrame(team_starters)
    out_path = os.path.join(PROJECT_ROOT, "data/starters_by_season.csv")
    starters_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(starters_df)} rows)")


if __name__ == "__main__":
    main()
