import pandas as pd
from pybaseball import playerid_lookup, statcast_batter
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('historical_pull_updated.csv')
combo_counts = df.groupby(["batter", "pitcher"]).size().reset_index(name="count")
result = df.groupby(["batter", "pitcher"])[["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play"]].sum().reset_index()
summary_stats = result.merge(combo_counts, on=["batter", "pitcher"])
max_count = summary_stats["count"].max()
most_frequent_combos = summary_stats[summary_stats["count"] == max_count]

print(summary_stats[(summary_stats['batter'] == 453568) & (summary_stats['pitcher'] == 605483)])

batter = statcast_batter(start_dt="2014-03-22", end_dt="2024-09-30", player_id='453568')
temp = batter[['game_date','events','description','home_team','away_team']][batter['pitcher'] == 605483]

unique_combos = batter[["events", "description"]].drop_duplicates()
print(unique_combos)
