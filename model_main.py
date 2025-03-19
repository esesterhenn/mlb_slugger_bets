import pandas as pd
from pybaseball import playerid_lookup, statcast_batter
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('historical_pull_updated.csv')
df['launch_angle_optimal'] = ((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).astype(int)
df['launch_speed_optimal'] = (df['launch_speed'] >= 95).astype(int)

def calculate_rolling_averages_and_sums(df,avg_cols,sum_cols,window,key_cols,date_col):
    # Convert game_date to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort the DataFrame
    df = df.sort_values(key_cols)
    
    # Set game_date as index
    df = df.set_index(date_col)

    group_cols = key_cols.copy()
    group_cols.remove(date_col)

    # Calculate rolling averages
    rolling_avg_df = (df.groupby(group_cols)[avg_cols]
                     .rolling(window, closed='both')
                     .mean()
                     .rename(columns={col: f'{col}_{window}_avg' for col in avg_cols})
                     )
    
    # Calculate rolling sums
    rolling_sum_df = (df.groupby(group_cols)[sum_cols]
                     .rolling(window, closed='both')
                     .sum()
                     .rename(columns={col: f'{col}_{window}_sum' for col in sum_cols})
                     )
    
    # Combine the results
    rolling_df = pd.concat([rolling_avg_df, rolling_sum_df], axis=1)
    rolling_df = rolling_df.reset_index()
    
    # Group by game_date and batter to get the last value for each day
    result_df = (rolling_df.groupby(key_cols)
                .agg({f'{col}_{window}_avg': 'last' for col in avg_cols}
                     | {f'{col}_{window}_sum': 'last' for col in sum_cols})
                .reset_index())
    
    # Reorder columns
    result_cols = (key_cols + 
                  [f'{col}_{window}_avg' for col in avg_cols] + 
                  [f'{col}_{window}_sum' for col in sum_cols])
    
    result_df = result_df[result_cols]

    result_df[f'AB_{window}'] = result_df[[f'single_{window}_sum', f'double_{window}_sum', f'triple_{window}_sum',f'home_run_{window}_sum',f'strikeout_{window}_sum',
                                           f'out_in_play_{window}_sum',f'field_error_{window}_sum']].sum(axis=1)
    result_df[f'AVG_{window}'] = result_df[[f'single_{window}_sum', f'double_{window}_sum', f'triple_{window}_sum',
                                            f'home_run_{window}_sum']].sum(axis=1)/result_df[f'AB_{window}']
    result_df[f'OBP_{window}'] = result_df[[f'single_{window}_sum', f'double_{window}_sum', f'triple_{window}_sum',f'home_run_{window}_sum',
                                            f'walk_{window}_sum']].sum(axis=1)/result_df[[f'AB_{window}',f'walk_{window}_sum',f'sac_fly_{window}_sum']].sum(axis=1)
    result_df[f'SR_{window}'] = (result_df[f'single_{window}_sum'] + result_df[f'double_{window}_sum'] * 2 + result_df[f'triple_{window}_sum'] * 3 + result_df[f'home_run_{window}_sum']*4)/result_df[f'AB_{window}']

    return result_df


def calculate_history_stats(df,avg_cols,sum_cols,key_cols,date_col):
    # Convert game_date to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort the DataFrame
    df = df.sort_values(key_cols)
    
    # Set game_date as index
    df = df.set_index(date_col)

    group_cols = key_cols.copy()
    group_cols.remove(date_col)

    # Calculate rolling averages
    rolling_avg_df = (df.groupby(group_cols)[avg_cols]
                     .mean())
    
    # Calculate rolling sums
    rolling_sum_df = (df.groupby(group_cols)[sum_cols]
                     .sum())
    
    # Combine the results
    result_df = pd.concat([rolling_avg_df, rolling_sum_df], axis=1)
    result_df = result_df.reset_index()

    result_df['AB'] = result_df[['single', 'double', 'triple','home_run','strikeout',
                                           'out_in_play','field_error']].sum(axis=1)
    result_df['AVG'] = result_df[['single', 'double', 'triple',
                                            'home_run']].sum(axis=1)/result_df['AB']
    result_df['OBP'] = result_df[['single', 'double', 'triple','home_run',
                                            'walk']].sum(axis=1)/result_df[['AB','walk','sac_fly']].sum(axis=1)
    result_df['SR'] = (result_df['single'] + result_df['double'] * 2 + result_df['triple'] * 3 + result_df['home_run']*4)/result_df['AB']

    return result_df

result_6d = calculate_rolling_averages_and_sums(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '6d',['game_date','batter'],'game_date')
result_29d = calculate_rolling_averages_and_sums(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '29d',['game_date','batter'],'game_date')
result_29d_pitch_arm = calculate_rolling_averages_and_sums(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '29d',['game_date','batter','p_throws'],'game_date')
batter_history_pitch_arm = calculate_history_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             ['game_date','batter','p_throws'],'game_date')
batter_pitcher_stats = calculate_history_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             ['game_date','batter','pitcher'],'game_date')
batter_ballpark_stats = calculate_history_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             ['game_date','batter','home_team'],'game_date')
test_batter_7d = result_6d[result_6d['batter'] == 668939]
test_batter_30d = result_29d[result_29d['batter'] == 668939]
test_batter_30d_pitch_arm = result_29d_pitch_arm[result_29d_pitch_arm['batter'] == 668939]
test_batter_history_pitch_arm = batter_history_pitch_arm[batter_history_pitch_arm['batter'] == 668939]
test_batter_pitcher_stats = batter_pitcher_stats[batter_pitcher_stats['batter'] == 668939]
test_batter_ballpark_stats = batter_ballpark_stats[batter_ballpark_stats['batter'] == 668939]


print(test_batter_ballpark_stats.head())

if not os.path.exists('test_data'):
    os.makedirs('test_data')

#test_batter_7d.to_csv('test_data/batter_info_7D.csv', index=False)
#test_batter_30d.to_csv('test_data/batter_info_30D.csv', index=False)

#print(result_29d[result_29d['batter'] == 668939].head())

'''
# Set a threshold for minimum at bats for each of these
# Last 7 Days: Batting average (>.300), at least 1 homerun (count number of home runs), all stats: Threshold is 15 at bats: DONE
# Last 30 days: Average launch angle, % of hits between 8 and 32 degrees?, launch speed > 95 mph: Threshold is 7 day x3: DONE
# Batter vs left handers and right handers (last 30 days or career?) all stats: Threshold is 30 at bats and career is 150: DONE
# Batter vs ballpark all stats (career) what if the team gets a new ballpark? (Potentially braves and rangers new stadiums): Threshold is 10 at bats
# Batter vs Pitcher: Batting average (>.300), at least 1 homerun (count number of home runs), all stats: Threshold is 5 at bats: DONE

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