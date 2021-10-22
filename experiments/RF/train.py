#!/usr/bin/env python

"""Using RandomForest classifier to perform
   text classification"""

import logging
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


DATA_DIR = '../../train-test-dev'
MODEL_DIR = "Saved_Models"
LOG_DIR = "Logs"


def load_data(dir):
    df = pd.read_csv(dir+'/train.csv')
    X = df['article'].ravel()
    Y = df['topic']

    return X,Y


def identity(x):
    # """Dummy function that just returns the input"""
    return x


def main():

    # Load training and test sets.
    X_train, Y_train = load_data(DATA_DIR)

    # Convert the texts to vectors. We use a dummy function as tokenizer
    # and preprocessor, since the texts are already preprocessed and
    # tokenized.
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

    #Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/baseline_rf.log",level=logging.INFO)

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
        joblib.dump(classifier, f"{MODEL_DIR}/ccp_alpha_{ccp_alpha}", compress=9)


if __name__ == "__main__":
    main()
