import os
import requests
from dotenv import load_dotenv
import pandas as pd
import config

BASE = "https://api.collegefootballdata.com"

load_dotenv()

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

os.makedirs('data', exist_ok=True)
recruits_df.to_csv('data/recruit_data.csv', index=False,  )