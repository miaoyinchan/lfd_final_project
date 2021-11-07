#!/usr/bin/env python

"""Use trained model to predict
   on test set, and save the
   predicted labels to csv file."""

import argparse

import joblib
import pandas as pd
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    LemmaTokenizer,
    tokenizer_pos_tag,
    tokenizer_ner_tag,
    create_arg_parser,
    set_args,
)


nlp = spacy.load("en_core_web_sm")


DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
OUTPUT_DIR = "../Output"


def load_data(filepath, testset):
    """Return test sets reading from csv files"""
    if testset=="24":
        df_test = pd.read_csv(f"{filepath}/test.csv")
    elif testset=="25":
        df_test = pd.read_csv(f"{filepath}/test_25th.csv")

    X = df_test["article"].ravel()
    Y = df_test["topic"]

    return X, Y


def main():
    args = create_arg_parser()

    # Obtain model/experiment name
    # based on the input setting
    experiment_name = set_args()

    # Load test set from train-test-dev folder
    X_test, Y_test = load_data(DATA_DIR, args.testset)

    # Load the saved model
    classifier = joblib.load(f"{MODEL_DIR}/{experiment_name}")

    # Test the model with test set
    Y_pred = classifier.predict(X_test)

    # Save results in dataframe
    df = pd.DataFrame()
    df["Test"] = Y_test
    df["Predict"] = Y_pred

    df.to_csv(f"{OUTPUT_DIR}/{experiment_name}.csv", index=False)


if __name__ == "__main__":
    main()
