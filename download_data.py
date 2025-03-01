import pandas as pd
from datetime import datetime, timedelta
from baseball_scraper import fangraphs
from baseball_id import Lookup

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

model_date = '2024-09-29'
model_start = (datetime.strptime(model_date,"%Y-%m-%d") - timedelta(days=3)).date()
model_end = (datetime.strptime(model_date,"%Y-%m-%d") - timedelta(days=1)).date()

print(model_start)
print(model_end)


# Constants for weighting
WEIGHT_WRC_PLUS = 0.40  # Last 30 Days wRC+
WEIGHT_LAST_7 = 0.50    # Last 7 days performance
WEIGHT_SPLITS = 0.10    # Full-season splits vs. LHP/RHP


'''
main_df = pd.DataFrame()
from pybaseball import playerid_lookup, statcast_batter
player_list = ['Aaron Judge','Ketel Marte']
for player_name in player_list:
    first_name = player_name.split(' ')[0]
    last_name = player_name.split(' ')[1]
    player = playerid_lookup(last_name, first_name)
    player_id = player.key_mlbam.iloc[0]
    df = statcast_batter(start_dt=model_start.strftime("%Y-%m-%d"), end_dt=model_end.strftime("%Y-%m-%d"), player_id=player_id)
    main_df = pd.concat([main_df, df], ignore_index=True)


from pybaseball import playerid_lookup, statcast_batter
main_df = pd.DataFrame()
for i in range(592450,592450+20,1):
    df = statcast_batter(start_dt=model_start.strftime("%Y-%m-%d"), end_dt=model_end.strftime("%Y-%m-%d"), player_id=i)
    main_df = pd.concat([main_df, df], ignore_index=True)
values_to_keep = ['home_run','walk','single','double','triple','strike_out']

from pybaseball import playerid_lookup, statcast_batter, statcast
main_df = statcast(start_dt=model_start.strftime("%Y-%m-%d"), end_dt=model_end.strftime("%Y-%m-%d"))
values_to_keep = ['home_run']
print(main_df[main_df['events'].isin(values_to_keep)])
'''

from pybaseball import playerid_lookup, statcast_batter, statcast
main_df = statcast(start_dt=model_start.strftime("%Y-%m-%d"), end_dt=model_end.strftime("%Y-%m-%d"))
values_to_keep = ['home_run', 'single', 'double', 'triple', 'walk', 'strikeout']
filtered_df = main_df[main_df['events'].isin(values_to_keep)]
print(filtered_df.head())

'''
from baseball_scraper import fangraphs
from baseball_id import Lookup
self.driver = webdriver.Chrome(options=options)
player_id = Lookup.from_names(['Aaron Judge']).iloc[0].fg_id
fangraphs.Scraper.instances()
fg = fangraphs.Scraper("Steamer (RoS)")
df = fg.scrape(player_id, scrape_as=fangraphs.ScrapeType.HITTER)
df.columns
'''

# Fetch Last 30 Days wRC+ Data
def fetch_last_30_days_wrc_plus():
    # Scrape the data from FanGraphs
    wrc_plus_df = fg.batting_stats(
        stat_columns=['wRC+'],
        start_date='2025-01-28',  # Adjust dates as needed
        end_date='2025-02-27'
    )
    return wrc_plus_df


# Fetch Last 7 Days Data
def fetch_last_7_days():
    # Implement your existing function to fetch last 7 days data
    pass

# Fetch Full-Season Splits Data
def fetch_splits():
    # Implement your existing function to fetch full-season splits data
    pass

# Process data
def process_data(wrc_plus_df, last_7_df, splits_df):
    # Merge and apply weighting
    merged_df = pd.merge(wrc_plus_df, last_7_df, on='player_id', how='left')
    merged_df = pd.merge(merged_df, splits_df, on='player_id', how='left')
    merged_df['final_score'] = (
        (merged_df['wRC+'] * WEIGHT_WRC_PLUS) +
        (merged_df['last_7_days_stat'] * WEIGHT_LAST_7) +  # Replace with actual column name
        (merged_df['split_stat'] * WEIGHT_SPLITS)           # Replace with actual column name
    )
    return merged_df

# Save to Excel with daily tracking
def save_to_excel(data):
    today = datetime.now().strftime("%Y-%m-%d")
    file_name = "mlb_slugger_bets.xlsx"
    with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='new') as writer:
        data.to_excel(writer, sheet_name=today, index=False)

# Main Execution
wrc_plus_data = fetch_last_30_days_wrc_plus()

last_7_data = fetch_last_7_days()
splits_data = fetch_splits()
if not wrc_plus_data.empty and not last_7_data.empty and not splits_data.empty:
    final_data = process_data(wrc_plus_data, last_7_data, splits_data)
    save_to_excel(final_data)
    print("Data saved successfully!")
else:
    print("Failed to retrieve all necessary data.")