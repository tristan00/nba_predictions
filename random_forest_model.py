import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import random
from datamanager import get_features

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
    print('accuracy: {0}, test size: {1}'.format(accuracy, len(test_x)))

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_features()
    run_model(train_x, train_y, test_x, test_y)
