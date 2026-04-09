"""
Fetch transfer portal data and compute net roster flow per team per year.

Portal year convention: year=Y contains players who transferred to play in season Y.
So portal_net features for year T describe the roster rebuild *before* season T.

Features per team-year:
  - portal_inbound_count: number of players transferring IN
  - portal_outbound_count: number of players transferring OUT
  - portal_net_count: inbound - outbound
  - portal_inbound_rating: sum of composite ratings for inbound players
  - portal_net_rating: inbound rating sum - outbound rating sum

Output: data/transfer_portal.csv
Coverage: 2018-present (portal formalised in 2018)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
from dotenv import load_dotenv
import pandas as pd
import config

BASE = "https://api.collegefootballdata.com"

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
TOKEN = os.getenv("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {TOKEN}"}

PORTAL_START = 2018  # Transfer portal formally introduced


def get_portal(year):
    """Fetch transfer portal entries for a given year."""
    print(f"  Fetching transfer portal for {year}...")
    r = requests.get(
        f"{BASE}/player/portal",
        params={"year": year},
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

    portal_start = max(PORTAL_START, config.season_start)
    for year in range(portal_start, config.season_end + 1):
        df = get_portal(year)
        if df.empty:
            print(f"  {year}: no data")
            continue

        # API returns camelCase; season field is named 'season' not 'year'
        df = df.rename(columns={"season": "year"})

        origin_col = "origin"
        dest_col = "destination"
        rating_col = "rating" if "rating" in df.columns else None

        if not origin_col or not dest_col:
            print(f"  {year}: unexpected columns {list(df.columns)}")
            continue

        df["rating_val"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0) if rating_col else 0

        # Inbound (destination team gains a player)
        if dest_col:
            inbound = (
                df.dropna(subset=[dest_col])
                .groupby(dest_col)
                .agg(
                    portal_inbound_count=("rating_val", "count"),
                    portal_inbound_rating=("rating_val", "sum"),
                )
                .rename_axis("team")
                .reset_index()
            )
        else:
            inbound = pd.DataFrame(columns=["team", "portal_inbound_count", "portal_inbound_rating"])

        # Outbound (origin team loses a player)
        if origin_col:
            outbound = (
                df.dropna(subset=[origin_col])
                .groupby(origin_col)
                .agg(
                    portal_outbound_count=("rating_val", "count"),
                    portal_outbound_rating=("rating_val", "sum"),
                )
                .rename_axis("team")
                .reset_index()
            )
        else:
            outbound = pd.DataFrame(columns=["team", "portal_outbound_count", "portal_outbound_rating"])

        merged = inbound.merge(outbound, on="team", how="outer").fillna(0)
        merged["portal_net_count"] = (
            merged["portal_inbound_count"] - merged["portal_outbound_count"]
        )
        merged["portal_net_rating"] = (
            merged["portal_inbound_rating"] - merged["portal_outbound_rating"]
        )
        merged["year"] = year
        all_records.append(merged)
        print(f"  {year}: {len(merged)} teams with portal activity")

    if not all_records:
        print("No transfer portal data fetched.")
        return

    result = pd.concat(all_records, ignore_index=True)
    result = result[["year", "team", "portal_inbound_count", "portal_outbound_count",
                     "portal_net_count", "portal_inbound_rating", "portal_net_rating"]]

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    result.to_csv(os.path.join(PROJECT_ROOT, "data/transfer_portal.csv"), index=False)
    print(f"\nSaved data/transfer_portal.csv ({len(result)} rows, "
          f"{result['year'].nunique()} seasons)")


if __name__ == "__main__":
    main()
