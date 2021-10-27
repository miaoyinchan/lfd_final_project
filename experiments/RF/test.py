#!/usr/bin/env python

import os
import argparse
import pandas as pd
import joblib


DATA_DIR = '../../train-test-dev'
MODEL_DIR = "Saved_Models"
OUTPUT_DIR = "Output"


def create_arg_parser():
    """
    Description:

    This method is an arg parser

    Return

    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    args = parser.parse_args()
    return args


def load_data(dir):
    df = pd.read_csv(dir+'/test.csv')
    X = df['article'].ravel()
    Y = df['topic']

    return X,Y


def main():
    args = create_arg_parser()
    # Load training and test sets.
    X_test, Y_test = load_data(DATA_DIR)

    if args.tfidf:
        experiment_name = "RF+Tf-idf"
    else:
        experiment_name = "RF+CV"

    # load saved RandomForest models
    ccp_alphas = [0.0, 0.01, 0.001, 0.0001]
    for ccp_alpha in ccp_alphas:
        classifier = joblib.load(f"{MODEL_DIR}/{experiment_name}_ccp_alpha_{ccp_alpha}")

        # Test the model with test set
        Y_pred = classifier.predict(X_test)

        #save results in dataframe
        df = pd.DataFrame()
        df['Test'] = Y_test
        df['Predict'] = Y_pred

        df.to_csv(f"{OUTPUT_DIR}/{experiment_name}_ccp_alpha_{ccp_alpha}.csv", index=False)


if __name__ == "__main__":
    main()
