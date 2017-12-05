import sqlite3
import traceback

with sqlite3.connect('nba.db') as conn:
    res = conn.execute('select * from player_game_contribution')
    for i in res:
        print(i)


