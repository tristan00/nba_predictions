import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import random
import datetime
import traceback
import time
from elo import rate_1vs1
import operator

starting_elo = 1000
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
elo_df = pd.DataFrame(columns=['team_name','date_played','elo'])

def get_team_features(team_id, player_df, results_df, history, game_date, result_characteristics_len = 10):
    team_bool = results_df['team_name'] == team_id

    team_info = player_df[player_df['team_name'] == team_id]
    #print(team_info.shape)

    if team_info.empty:
        raise Exception('No past games')

    team_info.fillna(0, inplace=True)
    team_info = team_info.select_dtypes(include=numerics)

    team_mean = team_info.mean(axis=0).values.tolist()
    team_skew = team_info.skew(axis=0).values.tolist()
    team_kurtosis = team_info.kurtosis(axis=0).values.tolist()
    team_median = team_info.median(axis=0).values.tolist()

    win_perc_list = []
    for h in history:
        recent_game = results_df[team_bool].sort_values('date_played', ascending=False).head(h)
        win_perc_list.append(recent_game['result'].mean())
    team_elo = [look_up_elo(results_df, team_id, game_date)]

    return team_elo + team_mean + team_skew  + team_kurtosis + team_median + win_perc_list
    #return team_elo + team_mean + win_perc_list


#generate 2 mirror sets of features to train
def get_features_for_game(g_id, location_dict, team_dict, game_results, players, history_length = (1, 5, 10, 25)):
    game_bool = game_results['g_id'] == g_id
    team_game_df = game_results[game_bool]
    game_date = game_results[game_results['g_id'] == g_id]['date_played'].values[0]
    game_datetime_date = datetime.datetime.strptime(game_date, '%Y-%m-%d').date()
    max_pre_game_period = datetime.datetime.strptime(game_date, '%Y-%m-%d') - datetime.timedelta(days=365)
    max_pre_game_period_str = str(max_pre_game_period.date())
    game_year = datetime.datetime.strptime(game_date, '%Y-%m-%d').date().year

    #get teams
    teams = list(team_game_df['team_name'])
    sorted_teams = sorted(teams)
    result_list = [i[0] for i in [team_game_df[team_game_df['team_name'] == i]['result'].values.tolist() for i in sorted_teams]]

    #get_players
    #get last game the team played, get the players there, if unavailable return none and exit the method
    #team 1:
    #past_game_bool = game_results['date_played'] < game_date

    is_player_df_before_game = players['date_played'] < game_date
    is_player_df_recent = players['date_played'] > max_pre_game_period_str
    is_result_df_before_game = game_results['date_played'] < game_date
    is_result_df_recent = game_results['date_played'] > max_pre_game_period_str


    pre_game_player_df = players[is_player_df_before_game & is_player_df_recent]
    pre_game_result_df = game_results[is_result_df_before_game & is_result_df_recent]
    #print(pre_game_player_df.shape)
    #print(pre_game_result_df.shape)

    try:
        team1_features = []
        team2_features=[]
        team1_features = get_team_features(sorted_teams[0], pre_game_player_df, pre_game_result_df, history_length, game_date)
        team1_features.append(players[players['team_name'] == sorted_teams[0]]['home_game'].values[0])
        team2_features = get_team_features(sorted_teams[1], pre_game_player_df, pre_game_result_df, history_length, game_date)
        team2_features.append(players[players['team_name'] == sorted_teams[1]]['home_game'].values[0])
    except:
        traceback.print_exc()
        return None

    #general features
    sorted_team_list = sorted([i for i in team_dict.keys()])
    #team_features = [one_hot_encode(sorted_team_list , i) for i in sorted_teams]
    #reversed_team_features = [i for i in reversed(team_features)]
    #team_features = sum(team_features, [])
    #reversed_team_features = sum(reversed_team_features, [])
    #team_features = one_hot_encode(sorted_team_list , list(game_results[game_results['g_id'] == g_id]['game_location']))
    #reversed_team_features = [i for i in reversed(team_features)]
    #location = location_dict[list(game_results[game_results['g_id'] == g_id]['game_location'])[0]]
    #get_play for largest value in history
    #get_past h games

    #sorted_most_recent_n_games = players[players['player_id'] == i].sort_values('date_played', ascending=False).head(h)
    general_features = [game_datetime_date.year, game_datetime_date.month]
    #reversed_general_features =  [game_datetime_date.year, game_datetime_date.month]
    output_features = result_list
    output_features_reversed = [i for i in reversed(result_list)]

    game_dict = dict()

    game_dict['team1_features'] = team1_features
    game_dict['team2_features'] = team2_features
    game_dict['general_features1'] = general_features
    game_dict['general_features2'] = general_features
    game_dict['result'] = output_features
    game_dict['result_reversed'] = output_features_reversed
    game_dict['teams'] = sorted_teams


    return (game_datetime_date, g_id, game_dict)

    #get game details
    #game_results.sort('date_played', ascending=False)
    #print(game_results.head(history_length))

def get_features():
    location_dict = get_location_mapping()
    team_dict = get_team_mapping()
    game_results, players = read_data()
    players.fillna(0, inplace=True)

    calculate_elo(game_results)
    start_time = time.time()

    g_ids = set(game_results['g_id'])
    output_dict = dict()
    feature_list_of_list = []
    for count, g in enumerate(g_ids):
        feature_list = get_features_for_game(g, location_dict, team_dict, game_results, players)
        if feature_list:
            game_date, g_id, game_dict = feature_list
            output_dict.setdefault(game_date, dict())
            output_dict[game_date][g_id] = game_dict
            feature_list_of_list.append(feature_list)
        print('{0} processed of {1}, output len: {2}, time:{3}'.format(count, len(g_ids), len(feature_list_of_list), (time.time()- start_time)/max(1, len(feature_list_of_list))))
    elo_df.to_pickle('models/elo_storage.pkl')
    return output_dict

def evaluate_predictions(predictions, results):
    correct = 0
    total = 0
    predictions = predictions.tolist()

    for i in range(len(results)):
        total += 1
        if (predictions[i][0] > predictions[i][1] and results[i][0] > results[i][1]) or \
                (predictions[i][0] < predictions[i][1] and results[i][0] < results[i][1]):
            correct += 1
    return correct/total

def run_model(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators = 128)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    accuracy = evaluate_predictions(pred, test_y)
    print('accuracy:', accuracy)

def one_hot_encode(array, element):
    output = [0 for i in array] + [0]
    if element in array:
        output[array.index(element)] = 1
    else:
        output[-1] = 1
    return output

def get_location_mapping():
    with sqlite3.connect('nba.db') as conn:
        location_map = dict()
        res = conn.execute('''select distinct location from game order by location''').fetchall()
        for i, j in enumerate(res):
            location_map[j[0]] = i
        return location_map

def get_team_mapping():
    with sqlite3.connect('nba.db') as conn:
        team_map = dict()
        res = conn.execute('''select distinct team_name from player_game_contribution order by team_name''').fetchall()
        for i, j in enumerate(res):
            team_map[j[0]] = i
        return team_map

def look_up_elo(result_df, team_name, date_played):
    global elo_df
    if ((elo_df['team_name'] == team_name) & (elo_df['date_played'] == date_played)).any():
        elo = elo_df[(elo_df['team_name'] == team_name) & (elo_df['date_played'] == date_played)]['elo'].values[0]
    else:
        last_game = result_df[(result_df['team_name'] == team_name) & (result_df['date_played']< date_played)].sort_values('date_played', ascending=False).head(1)
        if len(last_game) == 0:
            elo = starting_elo
        else:
            pre_game_result_df = result_df[result_df['date_played'] < list(last_game['date_played'])[0]]
            g_id = list(last_game['g_id'])[0]
            opponent_team = result_df[(result_df['team_name'] != team_name) & (result_df['g_id'] == g_id)]

            opponent_team_elo_before_last_game = look_up_elo(pre_game_result_df, opponent_team['team_name'].values[0], last_game['date_played'].values[0])
            elo_before_previous_game = look_up_elo(pre_game_result_df, team_name, last_game['date_played'].values[0])
            if last_game['result'].values[0] == 1:
                result = rate_1vs1(elo_before_previous_game, opponent_team_elo_before_last_game)
                elo = result[0]
            else:
                result = rate_1vs1(opponent_team_elo_before_last_game, elo_before_previous_game)
                elo = result[1]
        elo_dict = {'team_name':team_name,'date_played':date_played,'elo': elo}
        new_elo = pd.DataFrame([elo_dict])
        elo_df = elo_df.append(new_elo)
    return elo

def calculate_elo(results_df):
    global elo_df
    try:
        elo_df = pd.read_pickle('models/elo_storage.pkl')
    except:
        traceback.print_exc()
        results_df = results_df.sort_values('date_played', ascending=True)
        for i, j in results_df.iterrows():
            look_up_elo(results_df, j['team_name'], j['date_played'])
    print(elo_df.sort_values('date_played', ascending=False).head(50))


def read_data():
    with sqlite3.connect('nba.db') as conn:
        game_results = pd.read_sql('''select distinct game.g_id, team_game.team_name, team_game.result, team_game.date_played, team_game.game_location
        from game join team_game on game.g_id = team_game.g_id where team_game.date_played > ?''', conn, params=('1985-01-01',))

        team_contribution = pd.read_sql('''select *
                from team_total_performance ''', conn)
        return game_results, team_contribution

if __name__ == '__main__':
    get_features()

