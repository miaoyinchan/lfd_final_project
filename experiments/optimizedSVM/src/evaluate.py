#!/usr/bin/env python

import argparse
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from math import e
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nlp = spacy.load("en_core_web_sm")

OUTPUT_DIR = "../Output/"


def create_arg_parser():
    """
    Description:

    This method is an arg parser

    Return

    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    for i in range(1,12):
        parser.add_argument(
            f"-f{i}",
            f"--feature{i}",
            action="store_true",
            help=f"choose feature set {i} for the classifier",
        )
    for i in range(1,10):
        parser.add_argument(
            f"-tu{i}",
            f"--tuning{i}",
            action="store_true",
            help="tune hyperparameters for the best classifier",
        )
    parser.add_argument("-re", "--resampling", action="store_true",
                        help="Applying resampling strategy")
    for i in range(1,5):
        parser.add_argument(
            f"-fx{i}",
            f"--fixedsq{i}",
            action="store_true",
            help="tune hyperparameters for the best classifier",
        )

    args = parser.parse_args()
    return args


def tokenizer_pos_tag(doc):
    return [token.pos_ for token in nlp(doc)]


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def tokenizer_ner_tag(doc: str) -> [str]:
    return  [token.label_ for token in nlp(doc).ents]


def save_results(Y_test, Y_pred, experiment_name):
    ''' save results (accuracy, precision, recall, and f1-score)
    in csv file and plot confusion matrix '''

    test_report =  classification_report(Y_test, Y_pred, output_dict=True, digits=4)

    result = {"experiment": experiment_name}
    labels = [label for label in test_report.keys() if label.isupper()]

    for label in labels:
        report = test_report[label]
        result.update({
            f"precision-{label}": report['precision'],
            f"recall-{label}": report['recall'],
            f"f1-{label}": report['f1-score'],
        })


    result['accuracy'] = test_report['accuracy']
    result['macro f1-score'] = test_report['macro avg']['f1-score']

    try:
        df = pd.read_csv(OUTPUT_DIR+"results.csv")
        df = df.append(result, ignore_index=True)
        df.to_csv(OUTPUT_DIR+"results.csv",index=False)
    except FileNotFoundError:
        df = pd.DataFrame(result,index=[0])
        df.to_csv(OUTPUT_DIR+"results.csv",index=False)

    # save the confusion matrix of the model in png file
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(OUTPUT_DIR+"{}.png".format(experiment_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))


def main():
    args = create_arg_parser()
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

    C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

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

    if args.resampling:
        experiment_name = f"best_model_resampling"

    if args.fixedsq1:
        experiment_name = f"fixed_sq_{1}"

    if args.fixedsq2:
        experiment_name = f"fixed_sq_{10}"

    if args.fixedsq3:
        experiment_name = f"fixed_sq_{100}"

    if args.fixedsq4:
        experiment_name = f"fixed_sq_{1000}"

    output = pd.read_csv(f"{OUTPUT_DIR}{experiment_name}.csv")
    Y_test = output['Test']
    Y_predict = output['Predict']

    save_results(Y_test, Y_predict, f"{experiment_name}")


if __name__ == "__main__":
    main()
