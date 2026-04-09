"""
Fetch head coach history for all teams and compute tenure features.

For each team-year, outputs:
  - coach_tenure: years the current head coach has been at that school (0 = first year)
  - is_first_year_coach: 1 if tenure == 0, else 0

Output: data/coaches.csv
Columns: year, team, coach_name, coach_tenure, is_first_year_coach
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


def get_coaches(year):
    """Fetch all coaches active in a given year."""
    print(f"  Fetching coaches for {year}...")
    r = requests.get(
        f"{BASE}/coaches",
        params={"year": year},
        headers=headers,
    ).json()
    if isinstance(r, dict) and "message" in r:
        print(f"    API error: {r['message']}")
        return []
    return r or []


def main():
    # Collect all seasons across all years first so we can compute tenure
    # across the full history, not just within the window
    all_coach_seasons = []

    # Fetch a slightly wider window (back to 2010) so we can compute tenure
    # for coaches who started before 2015
    fetch_start = max(2010, config.season_start - 5)

    for year in range(fetch_start, config.season_end + 1):
        coaches = get_coaches(year)
        for coach in coaches:
            fname = coach.get("first_name", "")
            lname = coach.get("last_name", "")
            full_name = f"{fname} {lname}".strip()
            for season in coach.get("seasons", []):
                all_coach_seasons.append({
                    "coach_name": full_name,
                    "team": season.get("school"),
                    "year": season.get("year"),
                })

    if not all_coach_seasons:
        print("No coach data fetched.")
        return

    df = pd.DataFrame(all_coach_seasons).drop_duplicates(subset=["team", "year"])
    df = df.dropna(subset=["team", "year"])
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["team", "year"])

    # Compute tenure: years since current coach's first season at this school.
    # Strategy: find the first year a coach appears at a school; tenure = year - first_year.
    # When a new coach takes over, tenure resets to 0.
    records = []
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("year").reset_index(drop=True)
        tenure = 0
        prev_coach = None
        for _, row in grp.iterrows():
            if row["coach_name"] != prev_coach:
                tenure = 0
                prev_coach = row["coach_name"]
            else:
                tenure += 1
            records.append({
                "year": row["year"],
                "team": team,
                "coach_name": row["coach_name"],
                "coach_tenure": tenure,
                "is_first_year_coach": int(tenure == 0),
            })

    result = pd.DataFrame(records)
    # Filter to our model window
    result = result[result["year"].between(config.season_start, config.season_end)]

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    result.to_csv(os.path.join(PROJECT_ROOT, "data/coaches.csv"), index=False)
    print(f"\nSaved data/coaches.csv ({len(result)} rows, "
          f"{result['team'].nunique()} teams)")


if __name__ == "__main__":
    main()
