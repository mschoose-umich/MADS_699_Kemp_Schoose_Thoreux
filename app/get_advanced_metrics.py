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

start_year = config.season_start
end_year = config.season_end

# Conferences we are interested in
target_conferences = ["Big Ten", "SEC", "Big 12", "ACC"]

def get_sp_ratings(year):
    """Fetch SP+ ratings for a specific year."""
    print(f"Fetching SP+ ratings for {year}...")
    r = requests.get(
        f"{BASE}/ratings/sp",
        params={"year": year},
        headers=headers
    ).json()
    if isinstance(r, dict) and "message" in r:
        print(f"Error from API: {r['message']}")
        return pd.DataFrame()
    return pd.DataFrame(r)

def get_talent_composite(year):
    """Fetch Team Talent Composite for a specific year."""
    print(f"Fetching Team Talent Composite for {year}...")
    r = requests.get(
        f"{BASE}/talent",
        params={"year": year},
        headers=headers
    ).json()
    if isinstance(r, dict) and "message" in r:
        print(f"Error from API (Talent): {r['message']}")
        return pd.DataFrame()
    return pd.DataFrame(r)

def main():
    sp_all = []
    talent_all = []

    for year in range(start_year, end_year + 1):
        sp_df = get_sp_ratings(year)
        if not sp_df.empty:
            # Filter for target conferences
            sp_df = sp_df[sp_df['conference'].isin(target_conferences)]
            sp_all.append(sp_df)

        talent_df = get_talent_composite(year)
        if not talent_df.empty:
            talent_all.append(talent_df)

    if not sp_all or not talent_all:
        print("No data fetched.")
        return

    sp_final = pd.concat(sp_all, ignore_index=True)
    talent_final = pd.concat(talent_all, ignore_index=True)

    # Merge SP+ with Talent on year and team
    # Note: Ensure team names match between endpoints
    merged_df = pd.merge(
        sp_final,
        talent_final,
        on=['year', 'team'],
        how='left'
    )

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    merged_df.to_csv(os.path.join(PROJECT_ROOT, 'data/advanced_metrics.csv'), index=False)
    print(f"Saved merged data to data/advanced_metrics.csv. Total rows: {len(merged_df)}")

if __name__ == "__main__":
    main()
