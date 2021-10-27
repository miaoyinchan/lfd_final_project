#!/usr/bin/env python

"""Using RandomForest classifier to perform
   text classification"""

import argparse
import logging
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize


DATA_DIR = '../../train-test-dev'
MODEL_DIR = "Saved_Models"
LOG_DIR = "Logs"


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
    df = pd.read_csv(dir+'/train.csv')
    X = df['article'].ravel()
    Y = df['topic']

    return X,Y


def main():
    args = create_arg_parser()
    # Load training and test sets.
    X_train, Y_train = load_data(DATA_DIR)

    # Convert the texts to vectors. We use a dummy function as tokenizer
    # and preprocessor, since the texts are already preprocessed and
    # tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,3))
        experiment_name = "RF+Tf-idf"
    else:
        vec = CountVectorizer(tokenizer=word_tokenize,ngram_range=(1,3))
        experiment_name = "RF+CV"

    #Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/{experiment_name}.log",level=logging.INFO)

    ccp_alphas = [0.0, 0.01, 0.001, 0.0001]
    # set up pipeline
    for ccp_alpha in ccp_alphas:
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


if __name__ == "__main__":
    main()
