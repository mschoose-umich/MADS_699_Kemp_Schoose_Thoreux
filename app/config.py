season_start = 2015
season_end = 2025
recruit_start = 2009
recruit_end = 2025

# Prediction settings
prediction_years = [2026, 2027, 2028]
decay_rate = 0.80         # fallback if empirical AR(1) estimate unavailable
nil_start_year = 2022     # first full recruiting cycle under NIL
bootstrap_n = 200         # bootstrap samples for prediction intervals
confidence_level = 0.90   # 90% intervals

# Win probability model settings
train_start = 2021
upcoming_season = 2025

# Position mapping (raw position → standardized group)
position_mapping = {
    '?': None, 'ATH': None, 'KR': None, 'LS': None, 'PR': None,
    'C': 'OL', 'G': 'OL', 'OL': 'OL', 'OT': 'OL',
    'CB': 'DB', 'DB': 'DB', 'S': 'DB',
    'DE': 'DL', 'DL': 'DL', 'DT': 'DL', 'EDGE': 'DL', 'NT': 'DL',
    'ILB': 'LB', 'LB': 'LB', 'OLB': 'LB',
    'FB': 'FB', 'P': 'P', 'PK': 'PK',
    'QB': 'QB', 'RB': 'RB', 'TE': 'TE', 'WR': 'WR',
}

# Starters per position group
positions = {
    'QB': 1, 'RB': 1, 'FB': 1, 'WR': 2, 'TE': 1,
    'OL': 5, 'DL': 4, 'LB': 3, 'DB': 4, 'PK': 1, 'P': 1,
}