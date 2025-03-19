import pandas as pd
from pybaseball import playerid_lookup, statcast_batter
import psycopg2
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

'''
# Connect to the Neon PostgreSQL database
connection = psycopg2.connect(
    host='ep-restless-bread-a5n5z7ia-pooler.us-east-2.aws.neon.tech',
    database='Baseball_Bets',
    user='neondb_owner',
    password='npg_V5SZnUOecGh0',  # Replace with your password
    sslmode='require'
)

# Create a cursor object
cursor = connection.cursor()

# Query to select all rows from the table
query = "SELECT * FROM historical_data;"

# Execute the query and load the data into a DataFrame
df = pd.read_sql(query, connection)
'''
df = pd.read_csv('historical_pull_updated.csv')
#df = df.groupby(["batter", "game_date"])[["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error']].sum().reset_index()

def calculate_rolling_sums(df,pk_cols, columns_to_sum, number_of_days):
    pks = df[pk_cols].drop_duplicates()
    
    # Convert game_date to datetime if it isn't already
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Sort the data by batter and game_date
    df = df.sort_values(['batter', 'game_date'])
    # Create a rolling window partitioned by batter
    # Using a 7-day window ending on the game_date
    rolling_sums = (df.groupby('batter')
                   .apply(lambda x: x.set_index('game_date')[columns_to_sum]
                         .rolling(number_of_days, closed='right')
                         .sum())
                   .reset_index())
    
    # Merge the rolling sums back to the original dataframe
    result = pks.merge(rolling_sums, 
                     on=pk_cols, 
                     suffixes=('', '_' + number_of_days))
    
    return result

# Columns you want to sum
columns_to_sum = ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error']

# Calculate the rolling sums
seven_day_result = calculate_rolling_sums(df, ["batter", "game_date"],columns_to_sum, '7D')
seven_day_result['AB_7D'] = seven_day_result[['single_7D', 'double_7D', 'triple_7D','home_run_7D','strikeout_7D','out_in_play_7D','field_error_7D']].sum(axis=1)
seven_day_result['AVG_7D'] = seven_day_result[['single_7D', 'double_7D', 'triple_7D','home_run_7D']].sum(axis=1)/seven_day_result['AB_7D']
seven_day_result['OBP_7D'] = seven_day_result[['single_7D', 'double_7D', 'triple_7D','home_run_7D','walk_7D']].sum(axis=1)/seven_day_result[['AB_7D','walk_7D','sac_fly_7D']].sum(axis=1)
seven_day_result['SR_7D'] = (seven_day_result['single_7D'] + seven_day_result['double_7D'] * 2 + seven_day_result['triple_7D'] * 3 + seven_day_result['home_run_7D']*4)/seven_day_result['AB_7D']


thirty_day_result = calculate_rolling_sums(df, ["batter", "game_date"],columns_to_sum, '30D')
thirty_day_result['AB_30D'] = thirty_day_result[['single_30D', 'double_30D', 'triple_30D','home_run_30D','strikeout_30D','out_in_play_30D','field_error_30D']].sum(axis=1)
thirty_day_result['AVG_30D'] = thirty_day_result[['single_30D', 'double_30D', 'triple_30D','home_run_30D']].sum(axis=1)/thirty_day_result['AB_30D']
thirty_day_result['OBP_30D'] = thirty_day_result[['single_30D', 'double_30D', 'triple_30D','home_run_30D','walk_30D']].sum(axis=1)/thirty_day_result[['AB_30D','walk_30D','sac_fly_30D']].sum(axis=1)
thirty_day_result['SR_30D'] = (thirty_day_result['single_30D'] + thirty_day_result['double_30D'] * 2 + thirty_day_result['triple_30D'] * 3 + thirty_day_result['home_run_30D']*4)/thirty_day_result['AB_30D']


print(seven_day_result[seven_day_result['batter'] == 668939].head())
print(thirty_day_result[thirty_day_result['batter'] == 668939].head())


'''
print(df[(df['batter'] == 668939) & (df['pitcher'] == 621244)])
combo_counts = df.groupby(["batter", "pitcher"]).size().reset_index(name="n_games")
result = df.groupby(["batter", "pitcher"])[["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error']].sum().reset_index()
summary_stats = result.merge(combo_counts, on=["batter", "pitcher"])
summary_stats['AB'] = summary_stats[['single', 'double', 'triple','home_run','strikeout','out_in_play','field_error']].sum(axis=1)
summary_stats['AVG'] = summary_stats[['single', 'double', 'triple','home_run']].sum(axis=1)/summary_stats['AB']
# Need to fix OBP, not matching up. Potentially need to add hit by pitch as an option
summary_stats['OBP'] = summary_stats[['single', 'double', 'triple','home_run','walk']].sum(axis=1)/summary_stats[['AB','walk','sac_fly']].sum(axis=1)
summary_stats['SR'] = (summary_stats['single'] + summary_stats['double'] * 2 + summary_stats['triple'] * 3 + summary_stats['home_run']*4)/summary_stats['AB']


# Set a threshold for minimum at bats for each of these
# Last 7 Days: Batting average (>.300), at least 1 homerun (count number of home runs), all stats: Threshold is 15 at bats
# Last 30 days: Average launch angle, % of hits between 8 and 32 degrees?, launch speed > 95 mph: Threshold is 7 day x3
# Batter vs left handers and right handers (last 30 days or career?) all stats: Threshold is 30 at bats and career is 150
# Batter vs ballpark all stats (career) what if the team gets a new ballpark? (Potentially braves and rangers new stadiums): Threshold is 10 at bats
# Batter vs Pitcher: Batting average (>.300), at least 1 homerun (count number of home runs), all stats: Threshold is 5 at bats

# Code to pull the players with the most matchups
max_count = summary_stats["n_games"].max()
most_frequent_combos = summary_stats[summary_stats["n_games"] == max_count]

# Charlie Blackmon vs Blake Snell
print(summary_stats[(summary_stats['batter'] == 668939) & (summary_stats['pitcher'] == 621244)])

# Jose Berrios: 621244
# Adley Rutschman: 668939

player = playerid_lookup('Rutschman', 'Adley')
player_id = player.key_mlbam.iloc[0]
print(player_id)'
'''