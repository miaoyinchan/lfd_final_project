#!/usr/bin/env python

import os
import argparse
import pandas as pd
import joblib


DATA_DIR = '../../train-test-dev'
MODEL_DIR = "Saved_Models"
OUTPUT_DIR = "Output"


def load_data(dir):
    df = pd.read_csv(dir+'/test.csv')
    X = df['article'].ravel()
    Y = df['topic']

    return X,Y


def identity(x):
    # """Dummy function that just returns the input"""
    return x


def main():

    # Load training and test sets.
    X_test, Y_test = load_data(DATA_DIR)

    # load saved RandomForest models
    ccp_alphas = [0.0, 0.01, 0.001, 0.0001]
    for ccp_alpha in ccp_alphas:
        classifier = joblib.load(f"{MODEL_DIR}/ccp_alpha_{ccp_alpha}")

        # Test the model with test set
        Y_pred = classifier.predict(X_test)

        #save results in dataframe
        df = pd.DataFrame()
        df['Test'] = Y_test
        df['Predict'] = Y_pred

        df.to_csv(f"{OUTPUT_DIR}/ccp_alpha_{ccp_alpha}.csv", index=False)


if __name__ == "__main__":
    main()
