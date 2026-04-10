import pandas as pd
import config

train_start = config.train_start
upcoming_season = config.upcoming_season

teams = pd.read_csv('data/starters_by_season.csv')
games = pd.read_csv('data/game_data.csv')

# Fill null values with 0.0 indicating a player with no star rating
teams = teams.fillna(0.0)

# Remove teams with missing season data
missing_teams = []

for _, game in games.iterrows():
    season = game.season
    home = game.homeTeam
    away = game.awayTeam

    if len(teams[(teams['season'] == season) & (teams['team'] == home)]) < 1:
        missing_teams.append((season, home))

    if len(teams[(teams['season'] == season) & (teams['team'] == away)]) < 1:
        missing_teams.append((season,away))

missing_teams = set(missing_teams)
missing_df = pd.DataFrame(missing_teams, columns=["season", "team"])
games["home_key"] = list(zip(games["season"], games["homeTeam"]))
games["away_key"] = list(zip(games["season"], games["awayTeam"]))
games_clean = games[
    ~games["home_key"].isin(missing_teams) &
    ~games["away_key"].isin(missing_teams)
]

# Drop ties
games_clean = games_clean[games_clean.homePoints != games_clean.awayPoints]

# Save 2025 games for production dashboard
upcoming_games = games_clean[(games_clean.season == upcoming_season) & (games_clean.seasonType == 'regular')]
upcoming_games.to_csv('data/2025_games_clean.csv', index=False)

# Add target variable to games_clean
games_clean["target"] = (games_clean["homePoints"] > games_clean["awayPoints"]).astype(int)

# Engineer features from roster data
OL_group = [f'OL{i}' for i in range(1,6)]
WR_group = [f'WR{i}' for i in range(1,3)]
DL_group = [f'DL{i}' for i in range(1,5)]
LB_group = [f'LB{i}' for i in range(1,4)]
DB_group = [f'DB{i}' for i in range(1,5)]

teams['OL_avg'] = teams[OL_group].mean(axis=1)
teams['WR_avg'] = teams[WR_group].mean(axis=1)
teams['DL_avg'] = teams[DL_group].mean(axis=1)
teams['LB_avg'] = teams[LB_group].mean(axis=1)
teams['DB_avg'] = teams[DB_group].mean(axis=1)

# Save 2025 roster data for production dashboard
upcoming_rosters = teams[teams.season == upcoming_season]
upcoming_rosters.to_csv('data/2025_rosters.csv', index=False)

# Engineer features for training data
features = ['QB_diff',
            'OL_adv_1',
            'OL_adv_2',
            'WR_adv_1',
            'WR_adv_2',
            'RB_adv_1',
            'RB_adv_2',
            'TE_diff',
            'PK_diff',
            'P_diff',
            'QB1_vs_DB_2',
            'QB2_vs_DB_1',
            'PASS_SYNERGY_diff',
            'team_1_is_home']

# Helper function to calculate feature values for a given game
def build_matchup_row(year, team_1, team_2, team_1_is_home, team_1_won, teams):
    t1 = teams[(teams.season == year) & (teams.team == team_1)]
    t2 = teams[(teams.season == year) & (teams.team == team_2)]

    team_1_synergy = (
        2.0 * t1.QB1.iloc[0] +
        1.0 * t1.WR_avg.iloc[0] +
        0.8 * t1.TE1.iloc[0]
        ) / 3.8

    team_2_synergy = (
        2.0 * t2.QB1.iloc[0] +
        1.0 * t2.WR_avg.iloc[0] +
        0.8 * t2.TE1.iloc[0]
         ) / 3.8

    return {
        'QB_diff': t1.QB1.iloc[0] - t2.QB1.iloc[0],
        'OL_adv_1': t1.OL_avg.iloc[0] - t2.DL_avg.iloc[0],
        'OL_adv_2': t2.OL_avg.iloc[0] - t1.DL_avg.iloc[0],
        'WR_adv_1': t1.WR_avg.iloc[0] - t2.DB_avg.iloc[0],
        'WR_adv_2': t2.WR_avg.iloc[0] - t1.DB_avg.iloc[0],
        'RB_adv_1': t1.RB1.iloc[0] - t2.LB_avg.iloc[0],
        'RB_adv_2': t2.RB1.iloc[0] - t1.LB_avg.iloc[0],
        'TE_diff': t1.TE1.iloc[0] - t2.TE1.iloc[0],
        'PK_diff': t1.PK1.iloc[0] - t2.PK1.iloc[0],
        'P_diff': t1.P1.iloc[0] - t2.P1.iloc[0],
        'QB1_vs_DB_2': t1.QB1.iloc[0] - t2.DB_avg.iloc[0],
        'QB2_vs_DB_1': t2.QB1.iloc[0] - t1.DB_avg.iloc[0],
        'PASS_SYNERGY_diff': team_1_synergy - team_2_synergy,
        'team_1_is_home': team_1_is_home,
        'season': year,
        'team_1': team_1,
        'team_2': team_2,
        'target': team_1_won,
    }

# Helper function to build xy table
def build_xy(df, train=True):
    rows = []

    for year, home, away, home_won in df[['season', 'homeTeam', 'awayTeam', 'target']].itertuples(index=False):
        rows.append(build_matchup_row(year, home, away, 1, home_won, teams))
        if train:
            rows.append(build_matchup_row(year, away, home, 0, 1 - home_won, teams))

    Xy = pd.DataFrame(rows)

    return Xy

# Train split
train_years = [year for year in range(train_start, upcoming_season)]
train_raw_data = games_clean[games_clean.season.isin(train_years)]
Xy_train = build_xy(train_raw_data)

# Save training split
Xy_train.to_csv('data/Xy_train.csv', index=False)