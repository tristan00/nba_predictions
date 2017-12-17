import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import datetime
import random
from datamanager import get_features
import scraper
from scipy.stats import randint as sp_randint
import time
from sklearn import preprocessing

def evaluate_predictions(predictions, results):
    correct = 0
    total = 0
    predictions = predictions.tolist()

    for i in range(len(results)):
        total += 1
        if (predictions[i][0] > predictions[i][1] and results[i][0] > results[i][1]) or \
                (predictions[i][0] < predictions[i][1] and results[i][0] < results[i][1]):
            correct += 1
    return correct, total

def run_model(train_x, train_y, test_x, test_y):

    clf = RandomForestClassifier(n_estimators = 256)
    #print(np.squeeze(test_x))
    if len(test_x) > 1:
        train_x = np.nan_to_num(np.squeeze(np.array(train_x)))
        train_y = np.nan_to_num(np.squeeze(np.array(train_y)))
        test_x = np.nan_to_num(np.squeeze(np.array(test_x)))
        test_y = np.nan_to_num(np.squeeze(np.array(test_y)))
    elif len(test_x) == 1:
        #print()
        train_x = np.nan_to_num(np.squeeze(np.array(train_x)))
        train_y = np.nan_to_num(np.squeeze(np.array(train_y)))
        test_x = np.nan_to_num(np.squeeze(np.array(test_x)).reshape(1,-1))
        test_y = np.nan_to_num(np.squeeze(np.array(test_y)).reshape(1,-1))
    else:
        return 0, 0

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    correct, total = evaluate_predictions(pred, test_y)
    accuracy = correct/total
    print('accuracy: {0}, test size: {1}'.format(accuracy, len(test_x)))
    return correct, total

def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_features_for_test_date(feature_dict, test_date, test_future = False):
    train_game_dicts = [j for i, j in feature_dict.items() if i < test_date]
    if not test_future:
        test_game_dicts = [j for i, j in feature_dict.items() if i == test_date]
    else:
        test_game_dicts = [j for i, j in feature_dict.items() if i >= test_date]

    train_game_dicts = {k: v for d in train_game_dicts for k, v in d.items() }
    test_game_dicts = {k: v for d in test_game_dicts for k, v in d.items() }

    training_features = []
    for i, j in train_game_dicts.items():
        training_features.append([j['team1_features'] + j['team2_features'] + j['general_features1'], j['result']])
        training_features.append([j['team2_features'] + j['team1_features'] + j['general_features2'], j['result_reversed']])

    test_features = []
    for i, j in test_game_dicts.items():
        test_features.append([j['team1_features'] + j['team2_features'] + j['general_features1'], j['result']])

    train_x = [i[0] for i in training_features]
    train_y = [i[1] for i in training_features]
    test_x = [i[0] for i in test_features]
    test_y = [i[1] for i in test_features]
    return train_x, train_y, test_x, test_y

def incrementally_test(start_date, end_date, feature_dict):
    correct = 0
    total = 0
    date_list = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    for d in date_list:
        train_x, train_y, test_x, test_y = get_features_for_test_date(feature_dict, d)
        print(len(test_x))
        temp_correct, temp_total = run_model(train_x, train_y, test_x, test_y)
        correct += temp_correct
        total += temp_total
        print('date: {3}, correct: {0}, total: {1}, running accuracy: {2}'.format(correct, total, correct / max(1, total), d))

def test_all_future(start_date, features):
    correct = 0
    total = 0
    train_x, train_y, test_x, test_y = get_features_for_test_date(features, start_date, test_future=True)
    print(len(test_x))
    temp_correct, temp_total = run_model(train_x, train_y, test_x, test_y)
    correct += temp_correct
    total += temp_total
    print('date: {3}, correct: {0}, total: {1}, running accuracy: {2}'.format(correct, total, correct / max(1, total), start_date))

def tune_forest_hp(feature_dict):
    train_game_dicts = [j for i, j in feature_dict.items()]

    train_game_dicts = {k: v for d in train_game_dicts for k, v in d.items() }

    training_features = []
    for i, j in train_game_dicts.items():
        training_features.append([j['team1_features'] + j['team2_features'] + j['general_features1'], j['result']])
        training_features.append([j['team2_features'] + j['team1_features'] + j['general_features2'], j['result_reversed']])

    train_x = [i[0] for i in training_features]
    train_y = [[i[1][0]] for i in training_features]

    train_x = np.nan_to_num(np.squeeze(np.array(train_x)))
    train_y = np.nan_to_num(np.squeeze(np.array(train_y)))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)


    clf = RandomForestClassifier(n_estimators=128)
    param_grid = {"n_estimators":[256],
                 "max_depth": [10, 12],
                  "max_features": ['log2'],
                  "min_samples_split": [2, 10],
                  "min_samples_leaf": [1, 5],
                  "bootstrap": [True],
                  "criterion": ["entropy"]}

    grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=1)
    grid_search.fit(train_x, train_y)

    print(grid_search)
    report(grid_search.cv_results_)



if __name__ == '__main__':
    #scraper.main()
    feature_dict = get_features()
    tune_forest_hp(feature_dict)
    #february_1_2017 = datetime.date(2017, 1, 1)
    # december_5_2017 = datetime.date(2005, 1, 31)
    #today = datetime.date.today()
    #incrementally_test(february_1_2017, today, feature_dict)

