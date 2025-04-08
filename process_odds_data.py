import pandas as pd
from pybaseball import playerid_lookup, team_ids

odds_df = pd.read_csv('odds_api_hr_odds.csv')

team_df = pd.read_csv('odds_stats_team_map.csv')

odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"])

odds_df["game_num_in_day"] = (
    odds_df.sort_values("commence_time")
      .groupby(["home_team", "away_team", "batter_name", "game_date"])["commence_time"]
      .rank(method="dense")
      .astype(int)
)

odds_df = pd.merge(odds_df,team_df,left_on=['home_team'], right_on = ['ODDS_TEAM'], how="inner")
odds_df = odds_df.drop(columns=["ODDS_TEAM"])
odds_df = odds_df.rename(columns={
    'STATS_TEAM': 'stats_home_team'})
odds_df = pd.merge(odds_df,team_df,left_on=['away_team'], right_on = ['ODDS_TEAM'], how="inner")
odds_df = odds_df.drop(columns=["ODDS_TEAM"])
odds_df = odds_df.rename(columns={
    'STATS_TEAM': 'stats_away_team'})


odds_df[['first_name', 'last_name']] = odds_df['batter_name'].str.split(' ', n=1, expand=True)

all_player_odds = pd.DataFrame()

for index, row in odds_df.iterrows():
    player_data = playerid_lookup(row['last_name'], row['first_name'], fuzzy=True)[['key_mlbam']].drop_duplicates()
    player_data['game_date'] = row['game_date']
    player_data['game_num_in_day'] = row['game_num_in_day']
    player_data['home_team'] = row['stats_home_team']
    player_data['away_team'] = row['stats_away_team']
    player_data['sports_book'] = row['sports_book']
    player_data['point'] = row['point']
    player_data['over_price'] = row['over_price']
    player_data['under_price'] = row['under_price']
    player_data['batter_name'] = row['batter_name']
    all_player_odds = pd.concat([all_player_odds, player_data], axis=0, ignore_index=True)

all_player_odds['row_priority'] = (
    all_player_odds.groupby([
        'game_date',
        'game_num_in_day',
        'home_team',
        'away_team',
        'sports_book',
        'point',
        'over_price',
        'under_price',
        'batter_name'
    ], dropna=False)
    .cumcount() + 1
)

model_data = pd.read_csv('model_data.csv')

filter_df = pd.merge(model_data,all_player_odds,left_on=['game_date','game_num_in_day','home_team','batter'], right_on = ['game_date','game_num_in_day','home_team','key_mlbam'], how="inner")
filter_df = filter_df[['game_date','batter','game_num_in_day','game_pk','home_team','sports_book','point','over_price','under_price','batter_name','row_priority']]

book_mapping_df = (
    filter_df
    .sort_values('row_priority')  # Ensures lowest row_priority comes first
    .drop_duplicates(
        subset=['game_date', 'game_num_in_day', 'game_pk', 'home_team','sports_book','point','over_price','under_price','batter_name'],
        keep='first'
    )
)
book_mapping_df = book_mapping_df.drop(columns=["batter_name",'row_priority'])

pivoted_df = (
    book_mapping_df
    .pivot_table(
        index=['game_date', 'batter','game_num_in_day', 'game_pk', 'home_team'],
        columns='sports_book',
        values=['point', 'over_price', 'under_price'],
        aggfunc='first'  # In case there are duplicates, we'll just take the first occurrence
    )
)

pivoted_df.columns = [f'{col[1]}.{col[0]}' for col in pivoted_df.columns]
pivoted_df.columns = pivoted_df.columns.str.replace(r'[^a-zA-Z0-9._]', '', regex=True)
pivoted_df = pivoted_df.reset_index()

print(pivoted_df.head())
pivoted_df.to_csv('formatted_odds.csv',index=False)