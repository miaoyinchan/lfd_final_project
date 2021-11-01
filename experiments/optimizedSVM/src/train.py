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


nlp = spacy.load("en_core_web_sm")
csv.field_size_limit(sys.maxsize)

DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


def create_arg_parser():
    """
    Description:
    This method is an arg parser
    Return
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


def read_data(dataset):
    sentences = []
    labels = []
    with open(dataset, "r") as file:
        text = list(csv.reader(file, delimiter=","))
        for row in text[1:]:
            tokens = row[-2].strip().split()
            sentences.append(" ".join(tokens))
            labels.append(row[-1])
    return sentences, labels


def get_stop_words():
    """Get nltk stop words"""
    return stopwords.words("english")


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def tokenizer_pos_tag(doc):
    return [token.pos_ for token in nlp(doc)]


def tokenizer_ner_tag(doc):
    return [token.label_ for token in nlp(doc).ents]


def feature1_word_ngram(use_tfidf):
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
    vectorizer = CountVectorizer
    if use_tfidf:
        vectorizer = TfidfVectorizer
    return vectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        min_df=5,
        max_features=5000,
    )

    args = create_arg_parser()
    if args.tfidf:
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=(5, 5), min_df=5, max_features=5000
        )
    else:
        return CountVectorizer(
            analyzer="char_wb", ngram_range=(5, 5), min_df=5, max_features=5000
        )


def pipeline2_char_gram(use_tfidf):
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
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature2_char_gram(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2)])


def pipeline8_word_char_ngram(use_tfidf):
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
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature3_stop_words(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2)])


def pipeline9_word_ngram_stopwords(use_tfidf):
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
    f1 = feature1_word_ngram(use_tfidf)
    f2 = feature2_char_gram(use_tfidf)
    f3 = feature3_stop_words(use_tfidf)
    return FeatureUnion([("f1", f1), ("f2", f2), ("f3", f3)])


def pipeline10_word_char_ngram_stopwords():
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
    clf = LinearSVC(C=1.0, max_iter=1000000)
    vec = feature11_word_ngram_2_5(use_tfidf)
    pipeline = Pipeline(
        [
            ("vec", vec),
            ("cls", clf),
        ]
    )
    return pipeline


def saveModel(classifier, experiment_name):
    joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)


def main():
    args = create_arg_parser()

    # Load training and test sets.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

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
