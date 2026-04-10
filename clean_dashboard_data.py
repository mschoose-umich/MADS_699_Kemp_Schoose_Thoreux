import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import json
import config
import ast
from pygini import gini

pd.set_option('display.float_format', '{:.4f}'.format)

NIL_YEAR = 2021
POWER4 = {'SEC', 'Big Ten', 'Big 12', 'ACC'}

METRICS = ['averageRating', 'totalRating', 'commits', 'averageStars']

METRIC_LABELS = {
    'averageRating': 'Average Rating',
    'totalRating':   'Total Rating',
    'commits':       'Commits',
    'averageStars':  'Average Stars',
}

BASE = "https://api.collegefootballdata.com"

load_dotenv()

TOKEN = os.getenv("CFBD_API_KEY")
print(TOKEN)

headers = {"Authorization": f"Bearer {TOKEN}"}

def cfdb_api_get_range(endpoint, filename, start, end, extra_params=None):
    results = []
    for year in range(start, end + 1):
        params = {"year": year}
        if extra_params:
            params.update(extra_params)
        r = requests.get(f"{BASE}/{endpoint}", params=params, headers=headers).json()
        results.extend(r)

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{filename}.csv", index=False)
    return df

def cfdb_api_get_recruiting_groups(filename, start, end, extra_params=None):
    results = []
    for year in range(start, end + 1):
        params = {"startYear": year, "endYear": year}
        if extra_params:
            params.update(extra_params)
        r = requests.get(f"{BASE}/recruiting/groups", params=params, headers=headers).json()

        for record in r:
            record["year"] = year
        results.extend(r)

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{filename}.csv", index=False)

start = config.recruit_start
end   = config.recruit_end


cfdb_api_get_range('/recruiting/players', 'recruit_data', start, end, extra_params=None)
cfdb_api_get_range('/recruiting/teams', 'recruit_teams', start, end, extra_params=None)
cfdb_api_get_recruiting_groups('recruiting_groups', start, end)
cfdb_api_get_range("/teams", "teams", start, end)

def get_team_conferences(start_year=2009, end_year=2025) :
    headers = {"Authorization": f"Bearer {TOKEN}"}
    result  = {}

    for year in range(start_year, end_year + 1):
        resp = requests.get(
            f"{BASE}/teams/fbs",
            headers=headers,
            params={"year": year},
        )
        resp.raise_for_status()
        for team in resp.json():
            result[(team["school"], year)] = team["conference"]

    return result

def get_lat_lon(df, loc_colname, lat_colname='lat', lon_colname='lon'):
    df[lat_colname] = df[loc_colname].apply(lambda x: ast.literal_eval(x)['latitude'])
    df[lon_colname] = df[loc_colname].apply(lambda x: ast.literal_eval(x)['longitude'])
    return df

conf_map = get_team_conferences(start_year=2009, end_year=2025)

def clean_recruits():
    recruit_df = pd.read_csv('data/recruit_data.csv')

    recruit_df['conference'] = recruit_df.apply(lambda r: conf_map.get((r["committedTo"], r["year"])), axis=1)
    recruit_df['group']      = recruit_df['conference'].apply(
        lambda c: 'Power 4' if c in POWER4 else 'G5 / Ind.'
    )

    recruit_df = get_lat_lon(recruit_df, 'hometownInfo', 'home_lat', 'home_lon')
    recruit_df = recruit_df.dropna(subset=['home_lat', 'home_lon', 'committedTo'])
    recruit_df['stars'] = recruit_df['stars'].fillna(0).astype(int)

    return recruit_df

def clean_recruiting_groups():
    recruit_groups_df = pd.read_csv('data/recruiting_groups.csv').sort_values(
        by=['team', 'year', 'positionGroup'])
    teams_df = (pd.read_csv('data/teams.csv')
                .rename({'school': 'team'}, axis=1)
                .dropna()
                .drop_duplicates(subset=['team']))

    teams_df = get_lat_lon(teams_df, 'location')

    conf_year = (
        recruit_groups_df
        .groupby(['conference', 'positionGroup', 'year'])[METRICS]
        .mean().reset_index()
        .rename(columns={m: f'conf{m[0].upper()}{m[1:]}' for m in METRICS})
    )

    recruit_groups_agg = (
        recruit_groups_df
        .groupby(['team', 'conference', 'positionGroup', 'year'])[METRICS]
        .mean().reset_index()
        .rename(columns={m: f'school{m[0].upper()}{m[1:]}' for m in METRICS})
        .merge(conf_year, on=['conference', 'positionGroup', 'year'])
    )

    dashboard_df = recruit_groups_agg.merge(teams_df, on=['team', 'conference']).drop_duplicates(
        subset=['team', 'conference', 'positionGroup', 'year'])
    dashboard_df['logo'] = dashboard_df['logos'].apply(lambda x: ast.literal_eval(x)[0])

    for m in METRICS:
        sc = f'school{m[0].upper()}{m[1:]}'
        cc = f'conf{m[0].upper()}{m[1:]}'
        dashboard_df[f'{m}Diff'] = dashboard_df[sc] - dashboard_df[cc]

    return dashboard_df, teams_df

def pre_post_nil(recruit_df):
    recruit_df_gini = recruit_df.groupby(['committedTo', 'year'])['rating'].apply(
        lambda x: np.array(x, dtype=float)).reset_index()
    recruit_df_gini['gini'] = recruit_df_gini['rating'].apply(lambda r: gini(r))

    teams_df = pd.read_csv('data/recruit_teams.csv')

    recruit_df_agg = recruit_df.dropna(subset=['rating', 'committedTo']).groupby(
        ['year', 'committedTo', 'conference', 'group']).agg(
        avg_stars=('stars', 'mean'),
        num_commits=('stars', 'count'),
        avg_rating=('rating', 'mean'),
    ).reset_index()

    recruit_df_agg = recruit_df_agg.merge(
        recruit_df_gini, how='left', on=['year', 'committedTo']).drop(
        'rating',axis=1).rename({'committedTo': 'team'}, axis=1)

    recruit_df_agg = recruit_df_agg.merge(teams_df[['year', 'team', 'rank', 'points']], on=['year', 'team'], how='left')

    recruit_df_agg['period'] = recruit_df_agg['year'].apply(
        lambda y: f'pre_nil' if y < 2021 else f'post_nil'
    )

    pre_post_nil = recruit_df_agg.groupby(['team', 'period', 'group'])[
        ['avg_stars', 'num_commits', 'avg_rating', 'gini', 'rank', 'points']].mean().reset_index()

    wide = pre_post_nil.pivot(index=['team', 'group'], columns='period',
                              values=['avg_stars', 'num_commits', 'avg_rating', 'gini', 'rank', 'points'])
    wide.columns = [f'{col}_{period}' for col, period in wide.columns]
    wide = wide.reset_index()
    METRICS = ['avg_stars', 'num_commits', 'avg_rating', 'gini', 'rank', 'points']

    for col in METRICS:
        wide[f'diff_{col}'] = wide[f'{col}_post_nil'] - wide[f'{col}_pre_nil']

    TOP_N = 10
    rows = []
    for col in METRICS:
        sub = wide[['team', 'group', f'diff_{col}']].rename(columns={f'diff_{col}': 'delta'})
        sub["metric"] = col
        rows.append(sub.nlargest(TOP_N, 'delta').assign(category='Winner'))
        rows.append(sub.nsmallest(TOP_N, 'delta').assign(category='Loser'))

    long = pd.concat(rows, ignore_index=True)
    long['delta_label'] = long['delta'].apply(lambda x: f'{x:+.3f}')

    return long

dashboard_df, teams_df = clean_recruiting_groups()
recruits_df = clean_recruits()
pre_post_nil_df = pre_post_nil(recruits_df)

dashboard_df.to_csv('data/dashboard_df.csv', index=False)
teams_df.to_csv('data/teams_df.csv', index=False)
recruits_df.to_csv('data/recruit_df.csv', index=False)
pre_post_nil_df.to_csv('data/pre_post_nil_df.csv', index=False)






