import os
import requests
from dotenv import load_dotenv
import pandas as pd
import config

BASE = "https://api.collegefootballdata.com"

load_dotenv()

TOKEN = os.getenv("CFBD_API_KEY")

headers = {"Authorization": f"Bearer {TOKEN}"}

start_year = config.season_start
end_year = config.season_end

os.makedirs('data/rosters', exist_ok=True)

for season in range(start_year, end_year + 1):
    r = requests.get(
        f"{BASE}/roster",
        params={"year": season},
        headers=headers
        ).json()
    players = pd.DataFrame(r)
    csv_path = f'data/rosters/{season}_rosters.csv'
    players.to_csv(csv_path, index=False)