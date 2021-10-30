import re
import sys
import argparse
import random
import pandas as pd
import os
import csv
import nltk
import logging
import matplotlib.pyplot as plt
import numpy as np
import joblib
import spacy
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.util import ngrams, pr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
)


nlp = spacy.load("en_core_web_sm")
# nltk.download('wordnet')

DATA_DIR = '../../../train-test-dev'
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def read_data(dataset):
    sentences = []
    labels = []
    with open(dataset, "r") as file:
        csv.field_size_limit(sys.maxsize)
        text = list(csv.reader(file, delimiter=','))
        for row in text[1:]:
            tokens = row[-2].strip().split()
            sentences.append(" ".join(tokens[:513]))
            labels.append(row[-1])
    return sentences, labels


def saveModel(classifier,experiment_name ):
    #save model
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)

    except OSError as error:
        joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)


def main():
    # Load training and test sets.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")
    # print(X_train[:2])
    #Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log",level=logging.INFO)

    C_values = [1, 10, 100, 1000]
    for c in C_values:
        clf = LinearSVC(C=c, max_iter=1000000)
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,3),
            min_df=5, max_features=5000)
        pipeline = Pipeline(
            [("vec", vec),
             ("cls", clf),
            ]
        )
        pipeline.fit(X_train, Y_train)
        #save parameter in log
        logging.info(pipeline.get_params())
        saveModel(pipeline, f"fixed_sq_{c}")


if __name__ == "__main__":
    main()
