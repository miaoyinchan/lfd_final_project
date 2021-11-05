#!/usr/bin/env python

"""Tune Linear SVM model using
   TFIDF vectors, and word ngram
   range (1, 3) features to find
   the hyper-parameter."""

import csv
import logging
import sys

import joblib
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import (
    saveModel,
    read_data,
    train_model,
)


nlp = spacy.load("en_core_web_sm")
csv.field_size_limit(sys.maxsize)


DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def main():
    # Load training set.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

    # Tune model in a range of c value, and save models
    C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c in C_values:
        train_model(c, X_train, Y_train, f"tfidf_w_ngram_1_3_{c}")


if __name__ == "__main__":
    main()
