"""
Module requests game data from seasons based on dates defined in config.py as
season_start and season_end.
"""

import os
import requests
from dotenv import load_dotenv
import pandas as pd
import config

BASE = "https://api.collegefootballdata.com"

load_dotenv()

TOKEN = os.getenv("CFBD_API_KEY")

headers = {"Authorization": f"Bearer {TOKEN}"}

START_YEAR = config.season_start
END_YEAR = config.season_end

games_all = []

for season in range(START_YEAR, END_YEAR + 1):
    r = requests.get(
        f"{BASE}/games",
        params={"year": season},
        headers=headers
        ).json()
    games_all.extend(r)

games_df = pd.DataFrame(games_all)

os.makedirs('data', exist_ok=True)
games_df.to_csv('data/game_data.csv', index=False,  )
