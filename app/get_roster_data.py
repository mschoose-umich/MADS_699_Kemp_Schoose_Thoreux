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

os.makedirs(os.path.join(PROJECT_ROOT, 'data/rosters'), exist_ok=True)

for season in range(start_year, end_year + 1):
    r = requests.get(
        f"{BASE}/roster",
        params={"year": season},
        headers=headers
        ).json()
    players = pd.DataFrame(r)
    csv_path = os.path.join(PROJECT_ROOT, f'data/rosters/{season}_rosters.csv')
    players.to_csv(csv_path, index=False)