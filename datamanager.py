import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import random
import datetime
import traceback
import time

def get_team_features(team_id, player_df, results_df, history, result_characteristics_len = 10):
    team_bool = results_df['team_name'] == team_id
    last_n_games = results_df[team_bool].sort_values('date_played', ascending=False).head(5)
    if len(last_n_games.values.tolist())<1:
        return None
    previous_games_id = list(last_n_games['g_id'])
    previous_game_bool = player_df['g_id'].isin(previous_games_id)
    player_team_bool = player_df['team_name'] == team_id
    teams_players_last_game = set(player_df[previous_game_bool & player_team_bool]['player_id'])

    team_players_median_win_contribution = []
    team_players_median_loss_contribution = []

    for p in teams_players_last_game:
        try:
            player_bool = player_df['player_id'] == p
            win_bool = player_df['result'] == 1
            loss_bool = player_df['result'] == 0

            players_last_wins = player_df[player_bool & win_bool].sort_values('date_played', ascending=False).head(result_characteristics_len)
            players_last_losses = player_df[player_bool & loss_bool].sort_values('date_played', ascending=False).head(result_characteristics_len)

            win_df = players_last_wins.median(axis=0, numeric_only = True).to_frame()
            loss_df = players_last_losses.median(axis=0, numeric_only = True).to_frame()

            team_players_median_win_contribution.append(win_df)
            team_players_median_loss_contribution.append(loss_df)
        except:
            traceback.print_exc()


    team_players_median_win_contribution = pd.concat(team_players_median_win_contribution, axis = 1)
    team_players_mean_win_contribution = team_players_median_win_contribution.mean(axis=1).values.tolist() #average player
    team_players_skew_win_contribution = team_players_median_win_contribution.skew(axis=1).values.tolist()
    team_players_kurtosis_win_contribution = team_players_median_win_contribution.kurtosis(axis=1).values.tolist()
    team_players_max_win_contribution = team_players_median_win_contribution.max(axis=1).values.tolist()# best player

    team_players_median_loss_contribution = pd.concat(team_players_median_loss_contribution, axis = 1)
    team_players_mean_loss_contribution = team_players_median_loss_contribution.mean(axis=1).values.tolist() #average player
    team_players_skew_loss_contribution = team_players_median_win_contribution.skew(axis=1).values.tolist()
    team_players_kurtosis_loss_contribution = team_players_median_win_contribution.kurtosis(axis=1).values.tolist()
    team_players_max_loss_contribution = team_players_median_loss_contribution.max(axis=1).values.tolist()# best player

    win_perc_list = []
    for h in history:
        recent_game = results_df[team_bool].sort_values('date_played', ascending=False).head(h)
        win_perc_list.append(recent_game['result'].mean())

    return team_players_mean_win_contribution + team_players_max_win_contribution  + team_players_skew_win_contribution + \
           team_players_kurtosis_win_contribution + team_players_mean_loss_contribution + team_players_max_loss_contribution + \
           team_players_skew_loss_contribution + team_players_kurtosis_loss_contribution + win_perc_list


def get_general_features(player_df, results_df,  result_characteristics_len = 100):
    pass

#generate 2 mirror sets of features to train
def get_features_for_game(g_id, location_dict, team_dict, game_results, players, history_length = (1, 5, 10, 25)):
    game_bool = game_results['g_id'] == g_id
    team_game_df = game_results[game_bool]
    game_date = list(game_results[game_results['g_id'] == g_id]['date_played'])[0]
    game_datetime_date = datetime.datetime.strptime(game_date, '%Y-%m-%d').date()
    game_year = datetime.datetime.strptime(game_date, '%Y-%m-%d').date().year

    #get teams
    teams = list(team_game_df['team_name'])
    sorted_teams = sorted(teams)
    result_list = [i[0] for i in [team_game_df[team_game_df['team_name'] == i]['result'].values.tolist() for i in sorted_teams]]

    #get_players
    #get last game the team played, get the players there, if unavailable return none and exit the method
    #team 1:
    #past_game_bool = game_results['date_played'] < game_date
    try:
        team1_features = get_team_features(sorted_teams[0], players, game_results, history_length)
        team2_features = get_team_features(sorted_teams[0], players, game_results, history_length)
    except:
        return None

    #general features
    team_features = [team_dict[i] for i in sorted_teams]
    reversed_team_features = [i for i in reversed(team_features)]
    location = location_dict[list(game_results[game_results['g_id'] == g_id]['game_location'])[0]]
    #get_play for largest value in history
    #get_past h games

    #sorted_most_recent_n_games = players[players['player_id'] == i].sort_values('date_played', ascending=False).head(h)
    team_input_features = team1_features + team2_features
    team_input_features_reverse = team2_features + team1_features
    general_features = team_features + [location, game_year]
    reversed_general_features = reversed_team_features + [location, game_year]
    input_features = np.array(team_input_features + general_features)
    input_features_reversed = np.array(team_input_features_reverse + reversed_general_features)
    output_features = result_list
    output_features_reversed = [i for i in reversed([0, 1])]


    game_dict = dict()
    game_dict['team1_features'] = team1_features
    game_dict['team2_features'] = team2_features
    game_dict['general_features1'] = general_features
    game_dict['general_features2'] = reversed_general_features
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
    print(team_dict)
    game_results, players = read_data()
    players.fillna(0, inplace=True)
    start_time = time.time()

    g_ids = set(players['g_id'])
    output_dict = dict()
    feature_list_of_list = []
    for count, g in enumerate(g_ids):
        feature_list = get_features_for_game(g, location_dict, team_dict, game_results, players)
        if feature_list:
            game_date, g_id, game_dict = feature_list
            output_dict.setdefault(game_date, dict())
            output_dict[game_date][g_id] = game_dict
            feature_list_of_list.append(feature_list)
        print('{0} processed of {1}, output len: {2}, time:{3}'.format(count, len(g_ids), len(feature_list_of_list), (time.time()- start_time)/len(feature_list_of_list)))

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

def read_data():
    with sqlite3.connect('nba.db') as conn:
        game_results = pd.read_sql('''select distinct game.g_id, team_game.team_name, team_game.result, team_game.date_played, team_game.game_location
        from game join team_game on game.g_id = team_game.g_id''', conn)

        players = pd.read_sql('''select *
                from player_game_contribution''', conn)
        return game_results, players

if __name__ == '__main__':

    train_x, train_y, test_x, test_y = get_features()
    run_model(train_x, train_y, test_x, test_y)