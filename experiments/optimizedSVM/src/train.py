#!/usr/bin/env python

"""Training Linear SVM models on the full
   training set of COP meeting data
   with different features, and tuning
   these models to find the best one
   for binary text classification."""

import argparse
import csv
import logging
import sys

import joblib
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from utils import (
    saveModel,
    read_data,
    LemmaTokenizer,
    tokenizer_pos_tag,
    tokenizer_ner_tag,
)


nlp = spacy.load("en_core_web_sm")
csv.field_size_limit(sys.maxsize)


DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def create_arg_parser():
    """
    Description:
    This method is an arg parser
    Return:
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tfidf",
        action="store_true",
        help="Use the TF-IDF vectorizer instead of CountVectorizer",
    )
    for i in range(1, 12):
        parser.add_argument(
            f"-f{i}",
            f"--feature{i}",
            action="store_true",
            help=f"choose feature set {i} for the classifier",
        )

    args = parser.parse_args()
    return args


def get_stop_words():
    """Get nltk stop words"""
    return stopwords.words("english")


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


def feature1_word_ngram(use_tfidf):
    """Vectorize word ngram range(1,3)
       (feature set 1)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )


def pipeline1_word_ngram(use_tfidf):
    """Set pipeline using feature set 1"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature1_word_ngram(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature2_char_gram(use_tfidf):
    """Vectorize features character 5gram
       (feature set 2)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        min_df=5,
        max_features=5000,
    )


def pipeline2_char_gram(use_tfidf):
    """Set pipeline using feature set 2"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature2_char_gram(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature3_stop_words(use_tfidf):
    """Vectorize features word ngram range (1,3)
    excluding stop words (feature set 3)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        min_df=5,
        stop_words=get_stop_words(),
        max_features=5000,
    )


def pipeline3_stop_words(use_tfidf):
    """Set pipeline using feature set 3"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature3_stop_words(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature4_pos(use_tfidf):
    """Vectorize features using pos tag
       (feature set 4)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=tokenizer_pos_tag,
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )


def pipeline4_pos(use_tfidf):
    """Set pipeline using feature set 4"""
    clf = LinearSVC(C=1.0, max_iter=100000000)
    vec = feature4_pos(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature5_lemma(use_tfidf):
    """Vectorize features using lemmatokenizer
       (feature set 5)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=LemmaTokenizer(),
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )


def pipeline5_lemma(use_tfidf):
    """Set pipeline using feature set 5"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature5_lemma(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature6_ner(use_tfidf):
    """Vectorize features using NER tag
       (feature set 6)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=tokenizer_ner_tag,
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )


def pipeline6_ner(use_tfidf):
    """Set pipeline using feature set 6"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature6_ner(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature7_word_ngram_1_4(use_tfidf):
    """Vectorize features word ngram range(1,4)
       feature set 7"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 4),
        min_df=10,
        max_features=5000,
    )


def pipeline7_word_ngram_1_4(use_tfidf):
    """Set pipeline using feature set 7"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature7_word_ngram_1_4(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature8_word_char_ngram(use_tfidf):
    """Vectorize features combining word ngram
       range(1,3) and character 5gram
       (feature set 8)"""
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature2_char_gram(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2)])


def pipeline8_word_char_ngram(use_tfidf):
    """Set pipeline using feature set 8"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature8_word_char_ngram(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature9_word_ngram_stopwords(use_tfidf):
    """Union features word ngram range (1,3)
       and character 5gram (feature set 9)"""
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature3_stop_words(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2)])


def pipeline9_word_ngram_stopwords(use_tfidf):
    """Set up pipeline using features set 9"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature9_word_ngram_stopwords(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature10_word_char_ngram_stopwords(use_tfidf):
    """Union features word ngram range (1,3)
       and character 5gram and stop words
       (feature set 10)"""
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature2_char_gram(use_tfidf)
    f3 = feature3_stop_words(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2), ("f3", f3)])


def pipeline10_word_char_ngram_stopwords():
    """Set pipeline using feature set 10"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature10_word_char_ngram_stopwords()
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def feature11_word_ngram_2_5(use_tfidf):
    """Vectorize word ngram range(2,5)
       (feature set 11)"""
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        tokenizer=word_tokenize,
        ngram_range=(2, 5),
        min_df=10,
        max_features=5000,
    )


def pipeline11_word_ngram_2_5(use_tfidf):
    """Set pipeline using feature set 11"""
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature11_word_ngram_2_5(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def main():
    args = create_arg_parser()

    # Load training set.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

    # Select CV model and features set to train
    if args.feature1:
        pipeline = pipeline1_word_ngram(use_tfidf=args.tfidf)
        experiment_name = "cv_word_ngram"

    if args.feature2:
        pipeline = pipeline2_char_gram(use_tfidf=args.tfidf)
        experiment_name = "cv_char_ngram"

    if args.feature3:
        pipeline = pipeline3_stop_words(use_tfidf=args.tfidf)
        experiment_name = "cv_stop_words"

    if args.feature4:
        pipeline = pipeline4_pos(use_tfidf=args.tfidf)
        experiment_name = "cv_pos"

    if args.feature5:
        pipeline = pipeline5_lemma(use_tfidf=args.tfidf)
        experiment_name = "cv_lemma"

    if args.feature6:
        pipeline = pipeline6_ner(use_tfidf=args.tfidf)
        experiment_name = "cv_ner"

    if args.feature7:
        pipeline = pipeline7_word_ngram_1_4(use_tfidf=args.tfidf)
        experiment_name = "cv_word_ngram_1_4"

     # Select TFIDF model and features set to train
    if args.tfidf:
        if args.feature1:
            pipeline = pipeline1_word_ngram(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_ngram"

        if args.feature2:
            pipeline = pipeline2_char_gram(use_tfidf=args.tfidf)
            experiment_name = "tfidf_char_ngram"

        if args.feature3:
            pipeline = pipeline3_stop_words(use_tfidf=args.tfidf)
            experiment_name = "tfidf_stop_words"

        if args.feature4:
            pipeline = pipeline4_pos(use_tfidf=args.tfidf)
            experiment_name = "tfidf_pos"

        if args.feature5:
            pipeline = pipeline5_lemma(use_tfidf=args.tfidf)
            experiment_name = "tfidf_lemma"

        if args.feature6:
            pipeline = pipeline6_ner(use_tfidf=args.tfidf)
            experiment_name = "tfidf_ner"

        if args.feature7:
            pipeline = pipeline7_word_ngram_1_4(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_ngram_1_4"

        if args.feature8:
            pipeline = pipeline8_word_char_ngram(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_char_ngram"

        if args.feature9:
            pipeline = pipeline9_word_ngram_stopwords(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_ngram_stopwords"

        if args.feature10:
            pipeline = pipeline10_word_char_ngram_stopwords(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_char_ngram_stopwords"

        if args.feature11:
            pipeline = pipeline11_word_ngram_2_5(use_tfidf=args.tfidf)
            experiment_name = "tfidf_word_ngram_2_5"

    pipeline.fit(X_train, Y_train)
    # save parameters in log
    logging.info(pipeline.get_params())
    saveModel(pipeline, experiment_name)


if __name__ == "__main__":
    main()
