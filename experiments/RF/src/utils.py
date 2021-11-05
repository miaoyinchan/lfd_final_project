#!/usr/bin/env python

import argparse
import logging
import os
import pandas as pd
import joblib


DATA_DIR = '../../../train-test-dev'
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def load_data(directory):
    df = pd.read_csv(directory)
    X = df['article'].ravel()
    Y = df['topic']

    return X,Y


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
    parser.add_argument(
        "-b",
        "--bestmodel",
        action="store_true",
        help="Train model only with the hyper-parameter ccp_alpha=0.0",
    )
    args = parser.parse_args()
    return args
