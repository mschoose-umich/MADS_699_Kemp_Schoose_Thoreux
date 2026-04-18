"""
Fetch per-team, per-position-group recruiting quality from CFBD /recruiting/groups.

Provides continuous averageRating (0-1 composite scale) per position group,
which is finer-grained than star buckets (1-5 discrete).

Output: data/position_recruiting.csv
Wide format: year, team, rec_qb, rec_rb, rec_wr, rec_ol, rec_dl, rec_lb, rec_db
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

# Map API position group names → short column suffixes
POSITION_MAP = {
    "Quarterback": "rec_qb",
    "Running Back": "rec_rb",
    "Receiver": "rec_wr",
    "Offensive Line": "rec_ol",
    "Defensive Line": "rec_dl",
    "Linebacker": "rec_lb",
    "Defensive Back": "rec_db",
}


def get_position_recruiting(year):
    """Fetch position-group recruiting ratings for a year."""
    r = requests.get(
        f"{BASE}/recruiting/groups",
        params={"startYear": year, "endYear": year},
        headers=headers,
    ).json()
    if isinstance(r, dict) and "message" in r:
        print(f"    API error: {r['message']}")
        return pd.DataFrame()
    if not r:
        return pd.DataFrame()
    return pd.DataFrame(r)


def main():
    all_records = []

    for year in range(config.season_start, config.season_end + 1):
        print(f"  Fetching position recruiting for {year}...")
        df = get_position_recruiting(year)

        if df.empty:
            print(f"  {year}: no data")
            continue

        year_df = df
        year_df = year_df.drop_duplicates(subset=["team", "positionGroup"])

        # Pivot: one row per team, one column per position group
        pivot = year_df.pivot_table(
            index="team",
            columns="positionGroup",
            values="averageRating",
            aggfunc="mean",
        ).reset_index()

        # Rename columns using our position map; skip unmapped (e.g. "All Positions")
        renamed = {"team": "team"}
        for api_name, col_name in POSITION_MAP.items():
            if api_name in pivot.columns:
                renamed[api_name] = col_name

        pivot = pivot.rename(columns=renamed)
        keep = ["team"] + [c for c in POSITION_MAP.values() if c in pivot.columns]
        pivot = pivot[keep].copy()
        pivot["year"] = year
        all_records.append(pivot)

    if not all_records:
        print("No position recruiting data fetched.")
        return

    result = pd.concat(all_records, ignore_index=True)
    col_order = ["year", "team"] + [c for c in POSITION_MAP.values() if c in result.columns]
    result = result[col_order]

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    result.to_csv(os.path.join(PROJECT_ROOT, "data/position_recruiting.csv"), index=False)
    print(f"\nSaved data/position_recruiting.csv ({len(result)} rows, "
          f"{result['year'].nunique()} seasons)")


if __name__ == "__main__":
    main()
