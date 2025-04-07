import pandas as pd
from pybaseball import playerid_lookup, team_ids

odds_df = pd.read_csv('odds_api_hr_odds.csv')


odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"])

odds_df["game_num_in_day"] = (
    odds_df.sort_values("commence_time")
      .groupby(["home_team", "away_team", "batter_name", "game_date"])["commence_time"]
      .rank(method="dense")
      .astype(int)
)

odds_df[['first_name', 'last_name']] = odds_df['batter_name'].str.split(' ', n=1, expand=True)

for index, row in odds_df.head(5).iterrows():
    player_data = playerid_lookup(row['last_name'], row['first_name'], fuzzy=True)[['key_mlbam']].drop_duplicates()
    player_data['game_date'] = row['game_date']
    player_data['home_team'] = row['home_team']
    player_data['away_team'] = row['away_team']
    player_data['sports_book'] = row['sports_book']
    print(player_data)


#model_data = pd.read_csv('model_data.csv')