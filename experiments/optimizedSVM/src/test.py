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


def load_data(directory):
    """Load data from csv file"""
    df = pd.read_csv(directory)
    X = df["article"].ravel()
    Y = df["topic"]

    return X, Y


def main():
    args = create_arg_parser()

    # Obtain model/experiment name
    # based on the input setting
    experiment_name = set_args()

    # Test with an unseen test set
    if args.testset:
        X_test, Y_test = load_data(args.testset)
    else:
        # Load test sets.
        X_test, Y_test = load_data(f"{DATA_DIR}/test.csv")

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
