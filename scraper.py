import requests
import sqlite3
def build_db():
    with sqlite3.connect('reddit.db') as conn:
        conn.execute('create table if not exists game (game_id TEXT, team_id TEXT, date_of_game date, win int)')
        conn.execute('create table if not exists player (player_id TEXT UNIQUE, dob TEXT, s_id TEXT, author TEXT, parent_id TEXT, body TEXT, score int, submitted_timestamp TEXT, edited int)')
        conn.execute('create table if not exists team (p_id TEXT UNIQUE, s_id TEXT, author TEXT, title TEXT, body TEXT, score int, timestamp text, edited int)')