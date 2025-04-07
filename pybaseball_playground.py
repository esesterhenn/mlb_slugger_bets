import pybaseball
from pybaseball import playerid_reverse_lookup, statcast_single_game, season_game_logs, schedule_and_record
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

player_ids = [545341]
test = playerid_reverse_lookup(player_ids, key_type='mlbam')

data = statcast_single_game(716890)
print(data.head())

data = schedule_and_record(1927, "NYY")
print(data.head())