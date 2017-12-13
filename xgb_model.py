import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import datetime
import random
from datamanager import get_features
#import XGBoost as xgb
import scraper

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

    clf = RandomForestClassifier(n_estimators = 128)
    #print(np.squeeze(test_x))
    if len(test_x) > 1:
        train_x = np.squeeze(np.array(train_x))
        train_y = np.squeeze(np.array(train_y))
        test_x = np.squeeze(np.array(test_x))
        test_y = np.squeeze(np.array(test_y))
    elif len(test_x) == 1:
        #print()
        train_x = np.squeeze(np.array(train_x))
        train_y = np.squeeze(np.array(train_y))
        test_x = np.squeeze(np.array(test_x)).reshape(1,-1)
        test_y = np.squeeze(np.array(test_y)).reshape(1,-1)
    else:
        return 0, 0

    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    correct, total = evaluate_predictions(pred, test_y)
    accuracy = correct/total
    print('accuracy: {0}, test size: {1}'.format(accuracy, len(test_x)))
    return correct, total

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


if __name__ == '__main__':
    #scraper.main()
    feature_dict = get_features()
    february_1_2017 = datetime.date(2017, 1, 1)
    #december_5_2017 = datetime.date(2005, 1, 31)
    today = datetime.date.today()
    correct = 0
    total = 0


    date_list = [february_1_2017 + datetime.timedelta(days=x) for x in range((today - february_1_2017).days + 1)]
    for d in date_list:
        train_x, train_y, test_x, test_y = get_features_for_test_date(feature_dict, d)
        print(len(test_x))
        temp_correct, temp_total = run_model(train_x, train_y, test_x, test_y)
        correct+= temp_correct
        total += temp_total
        print('date: {3}, correct: {0}, total: {1}, running accuracy: {2}'.format(correct, total, correct/total, d))
