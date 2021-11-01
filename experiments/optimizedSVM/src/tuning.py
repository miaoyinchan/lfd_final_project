import csv
import logging
import sys

import joblib
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

nlp = spacy.load("en_core_web_sm")
csv.field_size_limit(sys.maxsize)


DATA_DIR = "../../../train-test-dev"
MODEL_DIR = "../Saved_Models"
LOG_DIR = "../Logs"


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


def saveModel(classifier, experiment_name):
    joblib.dump(classifier, f"{MODEL_DIR}/{experiment_name}", compress=9)


def main():
    # Load training and test sets.
    X_train, Y_train = read_data(f"{DATA_DIR}/train.csv")

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)

    # Tune hyperparameters in this range of c value
    C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c in C_values:
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

        # save parameters in log
        logging.info(pipeline.get_params())
        saveModel(pipeline, f"tfidf_w_ngram_1_3_{c}")


if __name__ == "__main__":
    main()
