import sqlite3
import traceback
import pandas as pd

with sqlite3.connect('nba.db') as conn:
    players = pd.read_sql('''select *
                    from team_total_performance''', conn)
    print(players.shape)


