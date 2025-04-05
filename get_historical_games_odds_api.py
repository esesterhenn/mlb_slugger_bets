import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('historical_pull_updated.csv')
df['game_date'] = pd.to_datetime(df['game_date'])
filtered_dates = df[df['game_date'] >= '2023-05-03']['game_date'].dt.date.unique()
utc_timestamps = [f"{d}T12:00:00Z" for d in sorted(filtered_dates)]


API_KEY = '528525ae5ca197d3c2d3d7da51d45624'
sport = 'baseball_mlb'
date_format = 'iso'

#markets = 'batter_home_runs'
#regions = 'us'
#odds_format = 'decimal'
all_games = pd.DataFrame()

for date in utc_timestamps:
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events?apiKey={API_KEY}&date={date}'
    params = {
        'dateFormat': date_format
    }
    response = requests.get(url, params=params)
    data = response.json()['data']
    parsed_df = pd.DataFrame(data)[['id', 'commence_time', 'home_team', 'away_team']]
    all_games = pd.concat([all_games, parsed_df], axis=0, ignore_index=True)

all_games['game_date'] = pd.to_datetime(all_games['commence_time']).dt.tz_convert('US/Central').dt.strftime('%Y-%m-%d')
final_df = all_games.drop_duplicates()

final_df.to_csv('odds_api_games.csv',index=False)