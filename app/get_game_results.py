"""
Fetch all FBS game results (2015-2025).

Fetches ALL FBS games per year (no conference filter) to ensure the Bradley-Terry
game graph includes every team regardless of conference realignment. Teams that
switched conferences (e.g., Oklahoma to SEC in 2024) will be present in all years.

Output: data/game_results.csv
Columns: year, home_team, home_conference, away_team, away_conference,
         home_points, away_points, neutral_site
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import requests
from dotenv import load_dotenv
import pandas as pd
import config

BASE = "https://api.collegefootballdata.com"

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
TOKEN = os.getenv("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {TOKEN}"}


def get_games_for_year(year):
    """Fetch all completed FBS games for a given year (regular + postseason)."""
    print(f"  Fetching all FBS games for {year}...")
    frames = []
    for season_type in ("regular", "postseason"):
        r = requests.get(
            f"{BASE}/games",
            params={"year": year, "seasonType": season_type, "classification": "fbs"},
            headers=headers,
        ).json()
        if isinstance(r, dict) and "message" in r:
            print(f"    API error ({season_type}): {r['message']}")
            continue
        if r:
            frames.append(pd.DataFrame(r))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    all_games = []

    for year in range(config.season_start, config.season_end + 1):
        year_df = get_games_for_year(year)

        if year_df.empty:
            print(f"  No games found for {year}")
            continue

        # Deduplicate by game id
        if "id" in year_df.columns:
            year_df = year_df.drop_duplicates(subset="id")

        # Keep only completed games with scores
        # API returns camelCase column names
        year_df = year_df.rename(columns={
            "season": "year",
            "homeTeam": "home_team",
            "homeConference": "home_conference",
            "awayTeam": "away_team",
            "awayConference": "away_conference",
            "homePoints": "home_points",
            "awayPoints": "away_points",
            "neutralSite": "neutral_site",
            "startDate": "start_date",
            "seasonType": "season_type",
        })
        keep_cols = [
            "id", "year", "home_team", "home_conference",
            "away_team", "away_conference", "home_points", "away_points",
            "neutral_site", "start_date", "season_type",
        ]
        available = [c for c in keep_cols if c in year_df.columns]
        year_df = year_df[available].copy()
        year_df = year_df.dropna(subset=["home_points", "away_points"])
        year_df["home_points"] = year_df["home_points"].astype(int)
        year_df["away_points"] = year_df["away_points"].astype(int)

        all_games.append(year_df)
        print(f"  {year}: {len(year_df)} unique completed games")

    if not all_games:
        print("No game data fetched.")
        return

    result = pd.concat(all_games, ignore_index=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    result.to_csv(os.path.join(PROJECT_ROOT, "data/game_results.csv"), index=False)
    print(f"\nSaved data/game_results.csv ({len(result)} rows, "
          f"{result['year'].nunique()} seasons)")


if __name__ == "__main__":
    main()
