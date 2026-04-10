"""
Module uses merged roster and recruit data to identify rankings of "starters".
Starters are assumed to be the highest ranked player at each position.
"""

import pandas as pd
import config
from pathlib import Path

SEASON_START = config.season_start
SEASON_END = config.season_end

POSITION_MAPPING = config.position_mapping
POSITIONS = config.positions

team_starters = []
for season in range(SEASON_START, SEASON_END + 1):
    data = pd.read_csv(Path(f'data/merged_rosters/{season}_rosters.csv'))
    data['position'] = data['position'].replace(POSITION_MAPPING)
    teams = data['team'].unique()
    for team in teams:
        roster = data[data['team'] == team]
        starters = {'season': season,
                    'team': team,}
        for position, num_starters in POSITIONS.items():
            players_at_pos = roster[roster['position'] == position].sort_values(by=['stars'],
                                                                            ascending=False
                                                                            )
            for i in range(num_starters):
                try:
                    starters[f'{position}{i + 1}'] = players_at_pos['stars'].iloc[i]
                except:
                    starters[f'{position}{i + 1}'] = 0.0
        team_starters.append(starters)

starters_by_season = pd.DataFrame(team_starters)

starters_by_season.to_csv('data/starters_by_season.csv', index=False)



