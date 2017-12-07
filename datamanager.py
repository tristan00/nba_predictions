import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import random

#generate 2 mirror sets of features to train
def get_features_for_game(g_id, location_dict, team_dict, game_results, players, history_length = 10):
    game_bool = game_results['g_id'] == g_id
    team_game_df = game_results[game_bool]
    game_date = list(game_results[game_results['g_id'] == g_id]['date_played'])[0]

    #get teams
    teams = list(team_game_df['team_name'])
    sorted_teams = sorted(teams)
    result_list = [i[0] for i in [team_game_df[team_game_df['team_name'] == i]['result'].values.tolist() for i in sorted_teams]]

    #get_players
    #get last game the team played, get the players there, if unavailable return none and exit the method
    #team 1:
    #past_game_bool = game_results['date_played'] < game_date
    team_bool = game_results['team_name'] == sorted_teams[0]
    last_2_games = game_results[team_bool].sort_values('date_played', ascending=False).head(2)
    if len(last_2_games.values.tolist())<1:
        return None
    previous_game = last_2_games.iloc[[1]]
    previous_game_id = list(previous_game.head(1)['g_id'])[0]
    teams_players_last_game = set(players[(players['g_id'] == previous_game_id) & (players['team_name'] == sorted_teams[0])]['player_id'])
    player_stats = []
    for i in teams_players_last_game:
        sorted_most_recent_n_games = players[players['player_id'] == i].sort_values('date_played', ascending=False).head(history_length)
        player_stats.append(sorted_most_recent_n_games)
    if len(player_stats) == 0:
        return None
    concat_player_stats = pd.concat(player_stats)
    concat_player_stats.fillna('', inplace=True)
    mean_features1 = concat_player_stats.mean(numeric_only = True).values.tolist()
    median_features1 = concat_player_stats.median(numeric_only=True).values.tolist()

    team_bool = game_results['team_name'] == sorted_teams[1]
    last_2_games = game_results[team_bool].sort_values('date_played', ascending=False).head(2)
    if len(last_2_games.values.tolist())<1:
        return None
    previous_game = last_2_games.iloc[[1]]
    previous_game_id = list(previous_game.head(1)['g_id'])[0]
    teams_players_last_game = set(players[(players['g_id'] == previous_game_id) & (players['team_name'] == sorted_teams[1])]['player_id'])
    player_stats = []
    for i in teams_players_last_game:
        sorted_most_recent_n_games = players[players['player_id'] == i].sort_values('date_played', ascending=False).head(history_length)
        player_stats.append(sorted_most_recent_n_games)
    if len(player_stats) == 0:
        return None
    concat_player_stats = pd.concat(player_stats)
    concat_player_stats.fillna('', inplace=True)
    mean_features2 = concat_player_stats.mean(numeric_only = True).values.tolist()
    median_features2 = concat_player_stats.median(numeric_only=True).values.tolist()

    team_input_features =  mean_features1 + median_features1 + mean_features2 + median_features2
    team_input_features_reverse = mean_features2 + median_features2 + mean_features1 + median_features1

    #general features
    team_features = [team_dict[i] for i in sorted_teams]
    reversed_team_features = [i for i in reversed(team_features)]
    location = location_dict[list(game_results[game_results['g_id'] == g_id]['game_location'])[0]]
    general_features = team_features + [location]
    reversed_general_features = reversed_team_features + [location]
    input_features = np.array(team_input_features + general_features)
    input_features_reversed = np.array(team_input_features_reverse + reversed_general_features)
    print(result_list[0])
    output_features = np.array(result_list)
    output_features_reversed = np.array([i for i in reversed([0, 1])])

    return [[input_features, output_features], [input_features_reversed, output_features_reversed]]

    #get game details
    #game_results.sort('date_played', ascending=False)
    #print(game_results.head(history_length))

def get_features(game_results, players, test_size = .1):
    location_dict = get_location_mapping()
    team_dict = get_team_mapping()
    print(team_dict)
    unique_game_ids = dict()
    game_results, players = read_data()
    players.fillna(0, inplace=True)

    g_ids = set(players['g_id'])
    inputs = []
    for g in g_ids:
        feature_list = get_features_for_game(g, location_dict, team_dict, game_results, players)
        if feature_list is not None:
            inputs.extend(feature_list)

    random.shuffle(inputs)
    train_set = inputs[:-int(len(inputs)*test_size)]
    test_set = inputs[-int(len(inputs) * test_size):]
    train_x = np.array(i[0] for i in train_set)
    train_y = np.array(i[1] for i in train_set)
    test_x = np.array(i[0] for i in test_set)
    test_y = np.array(i[1] for i in test_set)

    return train_x, train_y, test_x, test_y

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
    game_results, players = read_data()
    train_x, train_y, test_x, test_y = get_features(game_results, players)
    run_model(train_x, train_y, test_x, test_y)