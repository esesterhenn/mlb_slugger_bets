import requests
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('odds_api_games.csv')

test_data = df.head()

API_KEY = '528525ae5ca197d3c2d3d7da51d45624'
sport = 'baseball_mlb'
date_format = 'iso'
markets = 'batter_home_runs'
regions = 'us'
odds_format = 'decimal'

for index, row in test_data.iterrows():
    snpashot_time = (pd.to_datetime(row['commence_time']) - pd.Timedelta(hours = 1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    id = row['id']
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{id}/odds?apiKey={API_KEY}'
    params = {
        'dateFormat': date_format,
        'odds_format': odds_format,
        'regions': regions,
        'markets': markets,
        'date': snpashot_time
    }
    response = requests.get(url, params=params)
    data = response.json()['data']
    print(data)