import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    TOKEN = os.getenv("CFBD_API_KEY")
    if not TOKEN:
        print("CFBD_API_KEY not found in environment.")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}
    BASE = "https://api.collegefootballdata.com"

    print("Fetching teams data...")
    r = requests.get(f"{BASE}/teams", headers=headers)
    if r.status_code != 200:
        print(f"Error fetching teams: {r.status_code} - {r.text}")
        return

    teams = r.json()
    records = []
    for t in teams:
        loc = t.get("location", {})
        records.append({
            "team": t.get("school"),
            "conference": t.get("conference"),
            "mascot": t.get("mascot"),
            "latitude": loc.get("latitude") if loc else None,
            "longitude": loc.get("longitude") if loc else None,
            "venue_name": loc.get("name") if loc else None,
        })
    
    df = pd.DataFrame(records)
    # Drop rows without coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, 'data/team_locations.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved team locations to {out_path} ({len(df)} records)")

if __name__ == "__main__":
    main()
