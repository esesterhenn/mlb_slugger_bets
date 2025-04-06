import requests
import pandas as pd
import json

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('odds_api_games.csv')


API_KEY = '528525ae5ca197d3c2d3d7da51d45624'
sport = 'baseball_mlb'
date_format = 'iso'
markets = 'batter_home_runs'
regions = 'us'
odds_format = 'decimal'


def extract_hr_odds(data):
    records = []

    game_id = data['id']
    commence_time = data['commence_time']
    home_team = data['home_team']
    away_team = data['away_team']

    for book in data['bookmakers']:
        sports_book = book['title']
        for market in book.get('markets', []):
            if market['key'] != 'batter_home_runs':
                continue

            last_update = market['last_update']
            outcomes = market['outcomes']
            
            # organize over/under by player
            player_lines = {}
            for outcome in outcomes:
                name = outcome['description']
                point = outcome.get('point')
                if name not in player_lines:
                    player_lines[name] = {'point': point, 'Over': None, 'Under': None}
                player_lines[name][outcome['name']] = outcome.get('price')

            # build rows
            for batter_name, line in player_lines.items():
                records.append({
                    'id': game_id,
                    'commence_time': commence_time,
                    'home_team': home_team,
                    'away_team': away_team,
                    'sports_book': sports_book,
                    'last_update': last_update,
                    'batter_name': batter_name,
                    'point': line['point'],
                    'over_price': line['Over'],
                    'under_price': line['Under']
                })

    df = pd.DataFrame(records)
    return df

all_games = pd.DataFrame()

for index, row in df.iterrows():
    print(row['game_date'])
    snpashot_time = (pd.to_datetime(row['commence_time']) - pd.Timedelta(hours = 2)).strftime('%Y-%m-%dT%H:%M:%SZ')
    id = row['id']
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{id}/odds?apiKey={API_KEY}'
    params = {
        'dateFormat': date_format,
        'odds_format': odds_format,
        'regions': regions,
        'markets': markets,
        'date': snpashot_time
    }
    try:
        response = requests.get(url, params=params)
        data = response.json().get('data',None)
        if data is None:
            print('No data pulled')
        else:
            parsed_df = extract_hr_odds(data)
            parsed_df['game_date'] = row['game_date']
            all_games = pd.concat([all_games, parsed_df], axis=0, ignore_index=True)
    except requests.exceptions.RequestException as e:
        print(f"Error with game {row['game_date']}: {e}")
    
    except ValueError as e:
        print(f"Error decoding JSON for game {row['game_date']}: {e}")
    
    except Exception as e:
        print(f"Unexpected error for game {row['game_date']}: {e}")

all_games.to_csv('odds_api_hr_odds.csv',index=False)