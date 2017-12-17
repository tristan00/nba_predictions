import random_forest_model
import gradient_boost_classifier
from datamanager import get_features
import datetime
import scraper

def main():
    #scraper.main()
    feature_dict = get_features()
    start_date = datetime.date(2016, 1, 1)
    #today = datetime.date(2017, 12, 7)
    today = datetime.date.today()
    #random_forest_model.incrementally_test(start_date, today, feature_dict)
    random_forest_model.test_all_future(start_date, feature_dict)
    #gradient_boost_classifier.incrementally_test(start_date, today, feature_dict)

if __name__ == '__main__':
    main()
