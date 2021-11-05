#!/usr/bin/env python

"""Training Linear SVM model on the
   fixed sequence lenght training data
   and tuning to find the hyper-parameter."""

import csv
import logging
import sys
import pandas as pd

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import BertTokenizer

from utils import saveModel, read_data


# nlp = spacy.load("en_core_web_sm")
csv.field_size_limit(sys.maxsize)
tz = BertTokenizer.from_pretrained("bert-base-cased")


DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def bert_tokenizer(articles):
    tokenized_articles = []
    for article in articles:
        tokenized_article = str(tz.tokenize(article)[:512])
        tokenized_articles.append(tokenized_article)

    return tokenized_articles


def main():
    # Load training set.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Tokenize X_train with Bert tokenizer.
    X_train = bert_tokenizer(X_train)
    print(X_train[:1])

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

    # Tune model with to find the best C values.
    C_values = [1, 10, 100, 1000]
    for c in C_values:
        clf = LinearSVC(C=c, max_iter=1000000)
        vec = TfidfVectorizer(ngram_range=(1, 3),lowercase=False)
        pipeline = Pipeline(
            [
                ("vec", vec),
                ("cls", clf),
            ]
        )
        pipeline.fit(X_train, Y_train)

        # save parameter in log
        logging.info(pipeline.get_params())
        saveModel(pipeline, f"fixed_sq_{c}")


if __name__ == "__main__":
    main()
