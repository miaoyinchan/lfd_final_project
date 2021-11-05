#!/usr/bin/env python

import os
import argparse
import pandas as pd
import joblib

from utils import load_data, create_arg_parser


DATA_DIR = '../../../train-test-dev'
MODEL_DIR = "../Saved_Models"
OUTPUT_DIR = "../Output"


def print_predicted_labels_to_file(classifier, X_test, Y_test, experiment_name, ccp_alpha):
    # Predict labels using model
    Y_pred = classifier.predict(X_test)

    # Save results in dataframe
    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred
    df.to_csv(f"{OUTPUT_DIR}/{experiment_name}_ccp_alpha_{ccp_alpha}.csv", index=False)


def main():
    args = create_arg_parser()
    # Load training and test sets.
    X_test, Y_test = load_data(f"{DATA_DIR}/test.csv")

    if args.tfidf:
        experiment_name = "RF+Tf-idf"
    else:
        experiment_name = "RF+CV"

    # Select this option, only labels predicted by
    # model with  hyperparameter ccp_alpha=0.0 printed
    if args.bestmodel:
        classifier = joblib.load(f"{MODEL_DIR}/{experiment_name}_ccp_alpha_0.0")
        print_predicted_labels_to_file(classifier, X_test, Y_test, experiment_name, 0.0)

    else:
        # Print predicted labels of each model with different alpha
        ccp_alphas = [0.0, 0.01, 0.001, 0.0001]
        for ccp_alpha in ccp_alphas:
            classifier = joblib.load(f"{MODEL_DIR}/{experiment_name}_ccp_alpha_{ccp_alpha}")
            print_predicted_labels_to_file(classifier, X_test, experiment_name, ccp_alpha)


if __name__ == "__main__":
    main()
