import psycopg2
import pandas as pd

# Replace with your Neon database details
host = "ep-restless-bread-a5n5z7ia-pooler.us-east-2.aws.neon.tech"
dbname = "Baseball_Bets"
user = "neondb_owner"
password = "npg_V5SZnUOecGh0"

# Establish connection to the Neon database
connection = psycopg2.connect(
    host=host,
    dbname=dbname,
    user=user,
    password=password
)

# Create a cursor object
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS historical_data;")
connection.commit()

create_table_query = """
CREATE TABLE IF NOT EXISTS historical_data (
    game_date DATE,
    batter INTEGER,
    pitcher INTEGER,
    p_throws VARCHAR(1),
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    row_num INTEGER,
    launch_speed FLOAT,
    launch_angle FLOAT,
    hit_distance_sc FLOAT,
    double INTEGER,
    field_error INTEGER,
    home_run INTEGER,
    out_in_play INTEGER,
    sac_fly INTEGER,
    single INTEGER,
    strikeout INTEGER,
    triple INTEGER,
    walk INTEGER
);
"""

# Execute the table creation query
cursor.execute(create_table_query)
connection.commit() 

# Using COPY to upload the data from CSV directly into the table
copy_query = """
COPY historical_data (game_date, batter, pitcher, p_throws, home_team, away_team, row_num, launch_speed, launch_angle, hit_distance_sc, double, field_error, home_run,
out_in_play, sac_fly, single, strikeout, triple, walk)
FROM stdin WITH CSV HEADER DELIMITER as ','
"""
with open('historical_pull_updated.csv', 'r') as f:
    cursor.copy_expert(sql=copy_query, file=f)

# Commit and close
connection.commit()
cursor.close()
connection.close()