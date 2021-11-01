import argparse
import csv
import logging
import sys
from collections import Counter

import joblib
import spacy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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
        "-fxre",
        "--fixsq",
        action="store_true",
        help="Use fixed sequences (512 tokens) for training.",
    )

    args = parser.parse_args()
    return args


def read_data(dataset, use_fixed_sequence=False):
    sentences = []
    labels = []

    with open(dataset, "r") as file:
        text = list(csv.reader(file, delimiter=","))
        for row in text[1:]:
            tokens = row[-2].strip().split()
            if use_fixed_sequence:
                tokens = tokens[:512]
            sentences.append(" ".join(tokens))
            labels.append(row[-1])

    return sentences, labels


def saveModel(classifier, experiment_name):
    joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)


def main():
    args = create_arg_parser()
    # Load training and test sets.
    X_train, Y_train = read_data(
        f"{DATA_DIR}/train.csv",
        use_fixed_sequence=args.fixsq,
    )

    # Encode target label to fit in resampling.
    # encoder = preprocessing.LabelEncoder()
    # Y_train_encoded = encoder.fit_transform(Y_train)

    # Summarize class distribution before sampling
    counter = Counter(Y_train)
    print("Class distribution before resampling", counter)

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

    # Classifer with optimized c value.
    clf = LinearSVC(C=1000, max_iter=1000000)

    # Tfidf vectorize best features.
    vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        min_df=5,
        max_features=5000,
    )
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

    experiment_name = "best_model_resampling"
    if args.fixsq:
        experiment_name = "best_model_resampling_fixsq"

    saveModel(clf, experiment_name)


if __name__ == "__main__":
    main()
