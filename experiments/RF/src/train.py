#!/usr/bin/env python

"""Using RandomForest classifier to perform
   binary text classification"""

import argparse
import logging
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize

from utils import load_data


DATA_DIR = '../../../train-test-dev'
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


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


def train_model(ccp_alpha, vec, X_train, Y_train, experiment_name):
    classifier = Pipeline(
            [
                ("vec", vec),
                ("cls", RandomForestClassifier(random_state=0, ccp_alpha=ccp_alpha)),
            ]
        )

    # Train the model with training set.
    classifier.fit(X_train, Y_train)

    #save parameter in log
    logging.info(classifier.get_params())

    #save model
    joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}_ccp_alpha_{ccp_alpha}", compress=9)


def main():
    args = create_arg_parser()
    # Load training set.
    X_train, Y_train = load_data(f"{DATA_DIR}/train.csv")

    # Convert the texts to tfidf or CV
    if args.tfidf:
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,3), max_features=5000)
        experiment_name = "RF+Tf-idf"
    else:
        vec = CountVectorizer(tokenizer=word_tokenize,ngram_range=(1,3), max_features=5000)
        experiment_name = "RF+CV"

    #Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/{experiment_name}.log",level=logging.INFO)

    # Select this option, only model with hyperparameter ccp_alpha=0.0 trained
    if args.bestmodel:
        train_model(0.0, vec, X_train, Y_train, "RF+Tf-idf")

    else:
        # Tune hyperparameters in this range of ccp_alphas
        ccp_alphas = [0.0, 0.01, 0.001, 0.0001]
        for ccp_alpha in ccp_alphas:
            train_model(ccp_alpha, vec, X_train, Y_train, experiment_name)


if __name__ == "__main__":
    main()
