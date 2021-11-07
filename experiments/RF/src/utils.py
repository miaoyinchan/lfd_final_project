#!/usr/bin/env python

import argparse
import logging
import os
import pandas as pd
import joblib


DATA_DIR = '../../../train-test-dev'
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def load_data(filepath):
    """Return test sets reading from csv files"""
    df_test = pd.read_csv(filepath)
    X = df_test["article"].ravel()
    Y = df_test["topic"]

    return X, Y


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
    parser.add_argument(
        "-ts",
        "--testset",
        default="24",
        type=str,
        help="define the test set. By default it uses "
             "the 24th meeting as test set. Input "
             " '25' to use the 25th meeting as test set."
    )
    args = parser.parse_args()
    return args
