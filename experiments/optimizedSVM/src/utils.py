#!/usr/bin/env python

import argparse
import joblib
import csv
import sys
import pandas as pd
import logging

import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


nlp = spacy.load("en_core_web_sm")


MODEL_DIR = "../Saved_Models"


def saveModel(classifier,experiment_name ):
    """Save the trained model"""
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)

    except OSError as error:
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)


def read_data(dataset):
    """Read a text file and return
    a list of texts and a list of
    labels (binary classes)"""
    sentences = []
    labels = []
    with open(dataset, "r") as file:
        text = list(csv.reader(file, delimiter=","))
        for row in text[1:]:
            tokens = row[-2].strip().split()
            sentences.append(" ".join(tokens))
            labels.append(row[-1])
    return sentences, labels


# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def tokenizer_pos_tag(doc):
    """Get POS tag using spacy"""
    return [token.pos_ for token in nlp(doc)]


def tokenizer_ner_tag(doc):
    """Get NER tag using spacy"""
    return [token.label_ for token in nlp(doc).ents]


def create_arg_parser():
    """
    Description:
    This method is an arg parser
    Return:
    This method returns a map with commandline
    parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tfidf",
        action="store_true",
        help="Use the TF-IDF vectorizer instead of CountVectorizer",
    )
    parser.add_argument(
        "-ts",
        "--testset",
        type=str,
        help="Input directory of an unseen testset "
             "(only used for test.py)",
    )
    for i in range(1, 12):
        parser.add_argument(
            f"-f{i}",
            f"--feature{i}",
            action="store_true",
            help=f"Call model trained on full training set "
                  "with feature set {i}",
        )
    for i in range(1, 10):
        parser.add_argument(
            f"-tu{i}",
            f"--tuning{i}",
            action="store_true",
            help="Call model trained on full training set "
                 "with different C value",
        )
    for i in range(1, 5):
        parser.add_argument(
            f"-fx{i}",
            f"--fixedsq{i}",
            action="store_true",
            help="Call model trained on fixed sequence length "
                 "training set with different C value",
        )
    parser.add_argument(
        "-ud",
        "--updownsamplingaug",
        action="store_true",
        help="Call model trained on up- and downsampling data",
    )
    for i in range(1, 5):
        parser.add_argument(
            f"-u{i}",
            f"--upsamplingaug{i}",
            action="store_true",
            help="Call model trained on augmentation "
                 "data with different c values",
        )

    args = parser.parse_args()
    return args


def set_args():
    args = create_arg_parser()

    # Selection feature set for cv model
    if args.feature1:
        experiment_name = "cv_word_ngram"

    if args.feature2:
        experiment_name = "cv_char_ngram"

    if args.feature3:
        experiment_name = "cv_stop_words"

    if args.feature4:
        experiment_name = "cv_pos"

    if args.feature5:
        experiment_name = "cv_lemma"

    if args.feature6:
        experiment_name = "cv_ner"

    if args.feature7:
        experiment_name = "cv_word_ngram_1_4"

    # Selection feature set for tfidf model
    if args.tfidf:
        if args.feature1:
            experiment_name = "tfidf_word_ngram"

        if args.feature2:
            experiment_name = "tfidf_char_ngram"

        if args.feature3:
            experiment_name = "tfidf_stop_words"

        if args.feature4:
            experiment_name = "tfidf_pos"

        if args.feature5:
            experiment_name = "tfidf_lemma"

        if args.feature6:
            experiment_name = "tfidf_ner"

        if args.feature7:
            experiment_name = "tfidf_word_ngram_1_4"

        if args.feature8:
            experiment_name = "tfidf_word_char_ngram"

        if args.feature9:
            experiment_name = "tfidf_word_ngram_stopwords"

        if args.feature10:
            experiment_name = "tfidf_word_char_ngram_stopwords"

        if args.feature11:
            experiment_name = "tfidf_word_ngram_2_5"

    # Select c value for the model
    C_values = [0.00001, 0.0001, 0.001, 0.01,
                0.1, 1, 10, 100, 1000]

    if args.tuning1:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[0]}"

    if args.tuning2:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[1]}"

    if args.tuning3:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[2]}"

    if args.tuning4:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[3]}"

    if args.tuning5:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[4]}"

    if args.tuning6:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[5]}"

    if args.tuning7:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[6]}"

    if args.tuning8:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[7]}"

    if args.tuning9:
        experiment_name = f"tfidf_w_ngram_1_3_{C_values[8]}"

    if args.fixedsq1:
        experiment_name = "fixed_sq_1"

    if args.fixedsq2:
        experiment_name = "fixed_sq_10"

    if args.fixedsq3:
        experiment_name = "fixed_sq_100"

    if args.fixedsq4:
        experiment_name = "fixed_sq_1000"

    # Select model trained with upsampling
    # technique and c value
    if args.upsamplingaug1:
        experiment_name = "model_upsampling_aug_1"

    if args.upsamplingaug2:
         experiment_name = "model_upsampling_aug_10"

    if args.upsamplingaug3:
         experiment_name = "model_upsampling_aug_100"

    if args.upsamplingaug4:
         experiment_name = "model_upsampling_aug_1000"

    # Select model trained with a combination
    # technique of up- and downsampling
    if args.updownsamplingaug:
         experiment_name = "model_updownsampling_aug"

    return experiment_name


def train_model(c, X_train, Y_train, experiment_name):
    """Set up pipeline, train, and save model,
       save parameter in log file"""
    clf = LinearSVC(C=c, max_iter=1000000)
    vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    pipeline.fit(X_train, Y_train)

    # Save parameters in log
    logging.info(pipeline.get_params())

    saveModel(pipeline, experiment_name)
