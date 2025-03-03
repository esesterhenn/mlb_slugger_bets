import pandas as pd
from pybaseball import playerid_lookup, statcast_batter
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('historical_pull_updated.csv')
combo_counts = df.groupby(["batter", "pitcher"]).size().reset_index(name="n_games")
result = df.groupby(["batter", "pitcher"])[["double", "home_run", "single", "strikeout", "triple", "walk","out_in_play",'sac_fly','field_error']].sum().reset_index()
summary_stats = result.merge(combo_counts, on=["batter", "pitcher"])
summary_stats['AB'] = summary_stats[['single', 'double', 'triple','home_run','strikeout','out_in_play','field_error']].sum(axis=1)
summary_stats['AVG'] = summary_stats[['single', 'double', 'triple','home_run']].sum(axis=1)/summary_stats['AB']
# Need to fix OBP, not matching up. Potentially need to add hit by pitch as an option
summary_stats['OBP'] = summary_stats[['single', 'double', 'triple','home_run','walk']].sum(axis=1)/summary_stats[['AB','walk','sac_fly']].sum(axis=1)
summary_stats['SR'] = (summary_stats['single'] + summary_stats['double'] * 2 + summary_stats['triple'] * 3 + summary_stats['home_run']*4)/summary_stats['AB']

# Code to pull the players with the most matchups
max_count = summary_stats["n_games"].max()
most_frequent_combos = summary_stats[summary_stats["n_games"] == max_count]

# Charlie Blackmon vs Blake Snell
print(summary_stats[(summary_stats['batter'] == 453568) & (summary_stats['pitcher'] == 605483)])
