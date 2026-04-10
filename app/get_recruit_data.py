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

start_year = config.recruit_start
end_year = config.recruit_end

recruits_all = []

for recruit_year in range(start_year, end_year + 1):
    r = requests.get(
        f"{BASE}/recruiting/players",
        params={"year": recruit_year},
        headers=headers
        ).json()
    recruits_all.extend(r)

recruits_df = pd.DataFrame(recruits_all)

os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
recruits_df.to_csv(os.path.join(PROJECT_ROOT, 'data/recruit_data.csv'), index=False)