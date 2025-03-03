import pandas as pd
from datetime import datetime, timedelta
import pybaseball
from pybaseball import statcast
import warnings

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=FutureWarning)
pybaseball.cache.enable()

def split_date_ranges(date_list):
    max_interval = timedelta(days=6 * 30)  # Approximate 6 months as 180 days
    new_date_list = []
    
    for start_str, end_str in date_list:
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        
        if end_date - start_date > max_interval:
            temp_start = start_date
            while temp_start < end_date:
                temp_end = min(temp_start + timedelta(days=3 * 30), end_date)  # Approximate 3 months as 90 days
                new_date_list.append([temp_start.strftime("%Y-%m-%d"), temp_end.strftime("%Y-%m-%d")])
                temp_start = temp_end + timedelta(days=1)  # New start date is 1 day after previous end date
        else:
            new_date_list.append([start_str, end_str])
    
    return new_date_list


final_df = pd.DataFrame()
df_list = []
values_to_keep = ['home_run', 'single', 'double', 'triple', 'walk', 'strikeout','sac_fly','field_error']
date_list = [["2014-03-22", "2014-09-28"], ["2015-04-05", "2015-11-01"], ["2016-04-03", "2016-11-02"], 
             ["2017-04-02", "2017-11-01"], ["2018-03-29", "2018-10-01"], ["2019-03-20", "2019-09-29"], 
             ["2020-07-23", "2020-09-27"], ["2021-04-01", "2021-10-03"], ["2022-04-07", "2022-10-05"], 
             ["2023-03-30", "2023-11-01"], ["2024-03-28", "2024-09-30"]]


for item in split_date_ranges(date_list):
    start_date = item[0]
    end_date = item[1]
    print(str(start_date) + '_' + str(end_date))
    # Fetch data for the date range
    main_df = statcast(start_dt=start_date, end_dt=end_date)
    
    main_df = main_df[main_df["events"].notna() & (main_df["events"] != "truncated_pa")]
    main_df["events"] = main_df["events"].replace("intent_walk", "walk")
    main_df["events"] = main_df["events"].apply(lambda x: x if x in values_to_keep else "out_in_play")

    # Filter only relevant events
    filtered_df = main_df[main_df['events'].isin(values_to_keep + ["out_in_play"])]
    # Select only necessary columns
    filtered_df = filtered_df[['game_date','batter','pitcher','events','p_throws','home_team','away_team']]
    # Group and reshape data
    df = filtered_df.groupby(['game_date','batter','pitcher','p_throws','home_team','away_team'] + ['events']).size().unstack(fill_value=0)
    df = df.reset_index()
    # Append the grouped DataFrame to the list
    df_list.append(df)

# Concatenate all DataFrames at once
final_df = pd.concat(df_list, ignore_index=True)
final_df.to_csv('historical_pull_updated.csv',index=False)