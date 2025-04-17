import pandas as pd
from pybaseball import playerid_lookup, statcast_batter
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('historical_pull_updated.csv')
df['launch_angle_optimal'] = ((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).astype(int)
df['launch_speed_optimal'] = (df['launch_speed'] >= 95).astype(int)

def calculate_window_stats(df,avg_cols,sum_cols,window,key_cols,date_col,agg_type):
    # Convert game_date to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort the DataFrame
    df = df.sort_values(key_cols)
    
    # Set game_date as index
    df = df.set_index(date_col)

    group_cols = key_cols.copy()
    group_cols.remove(date_col)

    if agg_type == 'rolling':
        avg_df = (df.groupby(group_cols)[avg_cols]
                     .rolling(window, closed='both')
                     .mean()
                     .rename(columns={col: f'{col}_avg' for col in avg_cols})
                     )
        sum_df = (df.groupby(group_cols)[sum_cols]
                     .rolling(window, closed='both')
                     .sum()
                     .rename(columns={col: f'{col}_sum' for col in sum_cols})
                     )
    if agg_type == 'expanding':
        avg_df = (df.groupby(group_cols)[avg_cols]
                     .expanding()
                     .mean()
                     .rename(columns={col: f'{col}_avg' for col in avg_cols})
                     )
        sum_df = (df.groupby(group_cols)[sum_cols]
                     .expanding()
                     .sum()
                     .rename(columns={col: f'{col}_sum' for col in sum_cols})
                     )
    
    # Combine the results
    combined_df = pd.concat([avg_df, sum_df], axis=1)
    combined_df = combined_df.reset_index()
    
    # Group by game_date and batter to get the last value for each day
    result_df = (combined_df.groupby(key_cols)
                .agg({f'{col}_avg': 'last' for col in avg_cols}
                     | {f'{col}_sum': 'last' for col in sum_cols})
                .reset_index())
    
    # Reorder columns
    result_cols = (key_cols + 
                  [f'{col}_avg' for col in avg_cols] + 
                  [f'{col}_sum' for col in sum_cols])
    
    result_df = result_df[result_cols]

    result_df[f'AB'] = result_df[[f'single_sum', f'double_sum', f'triple_sum',f'home_run_sum',f'strikeout_sum',
                                           f'out_in_play_sum',f'field_error_sum']].sum(axis=1)
    result_df[f'AVG'] = result_df[[f'single_sum', f'double_sum', f'triple_sum',
                                            f'home_run_sum']].sum(axis=1)/result_df[f'AB']
    result_df[f'OBP'] = result_df[[f'single_sum', f'double_sum', f'triple_sum',f'home_run_sum',
                                            f'walk_sum']].sum(axis=1)/result_df[[f'AB',f'walk_sum',f'sac_fly_sum']].sum(axis=1)
    result_df[f'SR'] = (result_df[f'single_sum'] + result_df[f'double_sum'] * 2 + result_df[f'triple_sum'] * 3 + result_df[f'home_run_sum']*4)/result_df[f'AB']
    result_df[f'single_ratio'] = result_df[f'single_sum']/result_df[f'AB']
    result_df[f'double_ratio'] = result_df[f'double_sum']/result_df[f'AB']
    result_df[f'triple_ratio'] = result_df[f'triple_sum']/result_df[f'AB']
    result_df[f'home_run_ratio'] = result_df[f'home_run_sum']/result_df[f'AB']
    result_df[f'strikeout_ratio'] = result_df[f'strikeout_sum']/result_df[f'AB']
    result_df[f'out_in_play_ratio'] = result_df[f'out_in_play_sum']/result_df[f'AB']
    result_df[f'field_error_ratio'] = result_df[f'field_error_sum']/result_df[f'AB']

    return result_df


def get_last_date(df,key_cols,date_col,new_date_col):
    group_cols = key_cols.copy()
    group_cols.remove(date_col)
    unique_dates = (
    df[key_cols]
    .drop_duplicates()
    .sort_values(by=key_cols)
    )
    unique_dates[new_date_col] = unique_dates.groupby(group_cols)[date_col].shift(1)
    return unique_dates

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

result_6d = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '6d',['game_date','batter'],'game_date','rolling')
result_6d = result_6d[(result_6d['AB'] >= 15)]

result_29d = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '29d',['game_date','batter'],'game_date','rolling')
result_29d = result_29d[(result_29d['AB'] >= 60)]

result_29d_pitch_arm = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '29d',['game_date','batter','p_throws'],'game_date','rolling')
result_29d_pitch_arm = result_29d_pitch_arm[(result_29d_pitch_arm['AB'] >= 30)]

batter_history_pitch_arm = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '',['game_date','batter','p_throws'],'game_date','expanding')
batter_history_pitch_arm = batter_history_pitch_arm[(batter_history_pitch_arm['AB'] >= 100)]

batter_pitcher_stats = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '',['game_date','batter','pitcher'],'game_date','expanding')
batter_pitcher_stats = batter_pitcher_stats[(batter_pitcher_stats['AB'] >= 5)]

batter_ballpark_stats = calculate_window_stats(df,['launch_speed', 'launch_angle','launch_angle_optimal','launch_speed_optimal'],
                                             ["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error'],
                                             '',['game_date','batter','home_team'],'game_date','expanding')
batter_ballpark_stats = batter_ballpark_stats[(batter_ballpark_stats['AB'] >= 10)]


first_row = df.groupby(["game_date", "batter",'game_num_in_day','game_pk']).last().reset_index()
first_pitcher = first_row[["game_date", "batter",'game_num_in_day','game_pk', "pitcher", "p_throws","home_team","batter_last_name","batter_first_name","pitcher_last_name","pitcher_first_name"]]
pred_df = df.groupby(["game_date", "batter",'game_num_in_day','game_pk'])["home_run"].sum().reset_index()

pred_df = pred_df.merge(get_last_date(df,['game_date','batter'],'game_date','last_game_date'), on=["batter", "game_date"], how="left")
key_df = pd.merge(first_pitcher,pred_df, on=["game_date","batter",'game_num_in_day','game_pk'], how = "inner")


key_df = key_df.merge(get_last_date(key_df,['game_date','batter','p_throws'],'game_date','p_throws_date'), on=['game_date','batter','p_throws'], how="left")
key_df = key_df.merge(get_last_date(key_df,['game_date','batter','pitcher'],'game_date','pitcher_date'), on=['game_date','batter','pitcher'], how="left")
key_df = key_df.merge(get_last_date(key_df,['game_date','batter','home_team'],'game_date','home_date'), on=['game_date','batter','home_team'], how="left")


model_df = pd.merge(key_df,result_6d,left_on=['last_game_date','batter'], right_on = ['game_date','batter'], how="left", suffixes=("","_7D"))
model_df = model_df.drop(columns=["game_date_7D"])
model_df = pd.merge(model_df,result_29d,left_on=['last_game_date','batter'], right_on = ['game_date','batter'], how="left", suffixes=("","_30D"))
model_df = model_df.drop(columns=["game_date_30D"])
model_df = pd.merge(model_df,result_29d_pitch_arm,left_on=['p_throws_date','batter','p_throws'], right_on = ['game_date','batter','p_throws'], how="left", suffixes=("","_30D_arm"))
model_df = model_df.drop(columns=["game_date_30D_arm"])
model_df = pd.merge(model_df,batter_history_pitch_arm,left_on=['p_throws_date','batter','p_throws'], right_on = ['game_date','batter','p_throws'], how="left", suffixes=("","_hist_arm"))
model_df = model_df.drop(columns=["game_date_hist_arm"])
model_df = pd.merge(model_df,batter_pitcher_stats,left_on=['pitcher_date','batter','pitcher'], right_on = ['game_date','batter','pitcher'], how="left", suffixes=("","_hist_pitcher"))
model_df = model_df.drop(columns=["game_date_hist_pitcher"])
model_df = pd.merge(model_df,batter_ballpark_stats,left_on=['home_date','batter','home_team'], right_on = ['game_date','batter','home_team'], how="left", suffixes=("","_hist_home"))
model_df = model_df.drop(columns=["game_date_hist_home"])


model_df.to_csv('model_data.csv', index=False)

'''
# Set a threshold for minimum at bats for each of these
# Last 7 Days: Batting average (>.300), at least 1 homerun (count number of home runs), all stats: Threshold is 15 at bats: DONE
# Last 30 days: Average launch angle, % of hits between 8 and 32 degrees?, launch speed > 95 mph: Threshold is 7 day x4: DONE
# Batter vs left handers and right handers (last 30 days or career?) all stats: Threshold is 30 at bats and career is 100: DONE
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