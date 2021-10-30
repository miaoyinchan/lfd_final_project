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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from collections import Counter


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
            sentences.append(" ".join(tokens))
            labels.append(row[-1])
    return sentences, labels


def saveModel(classifier,experiment_name ):
    #save model
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)

    except OSError as error:
        joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)


def pipeline_resampling():
    '''Set up pipeline with optimized c value, the best
       features, and applying oversampling and undersampling
       technique'''
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,3),
        min_df=5, max_features=5000)
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1.0)
    pipeline = Pipeline(
        [("vec", vec),
         ("over", over),
         ("under", under),
         ("cls", clf),
        ]
    )
    return pipeline

def main():
    # Load training and test sets.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Encode target label to fit in resampling.
    # encoder = preprocessing.LabelEncoder()
    # Y_train_encoded = encoder.fit_transform(Y_train)

    # Summarize class distribution before sampling
    counter = Counter(Y_train)
    print("Class distribution before resampling", counter)

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log",level=logging.INFO)

    # Classifer with optimized c value.
    clf = LinearSVC(C=1000, max_iter=1000000)

    # Tfidf vectorize best features.
    vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,3),
        min_df=5, max_features=5000)
    X_train_tfidf = vec.fit_transform(X_train)

    # Oversampling on features vectors of CLIMATE using SMOTE
    # to have a half of the samples of MISC.
    over = SMOTE(sampling_strategy=0.5, k_neighbors=5)
    # Undersampling on features vectors of MISC to have same amount
    # samples of CLIMATE.
    under = RandomUnderSampler(sampling_strategy=1)

    X_train_sm, Y_train_sm = over.fit_resample(X_train_tfidf, Y_train)
    counter = Counter(Y_train_sm)
    print("Class distribution after oversampling", counter)

    X_train_un, Y_train_un = under.fit_resample(X_train_sm, Y_train_sm)
    counter = Counter(Y_train_un)
    print("Class distribution after undersampling", counter)

    clf.fit(X_train_un, Y_train_un)

    # Save parameters in log
    logging.info(clf.get_params())
    saveModel(clf, f"best_model_resampling")


if __name__ == "__main__":
    main()
