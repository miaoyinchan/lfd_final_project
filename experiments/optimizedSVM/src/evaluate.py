#!/usr/bin/env python

""" Evaluate the trained models
    and save the performance scores
    to csv file."""


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
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils import (
    LemmaTokenizer,
    tokenizer_pos_tag,
    tokenizer_ner_tag,
    create_arg_parser,
    set_args,
    train_model,
)

nlp = spacy.load("en_core_web_sm")

OUTPUT_DIR = "../Output/"


def save_results(Y_test, Y_pred, experiment_name):
    """Save results (accuracy, precision, recall, and f1-score)
    in csv file and plot confusion matrix"""

    test_report = classification_report(
        Y_test,
        Y_pred,
        output_dict=True,
        digits=4,
    )

    result = {"experiment": experiment_name}
    labels = [label for label in test_report.keys() if label.isupper()]

    for label in labels:
        report = test_report[label]
        result.update(
            {
                f"precision-{label}": report["precision"],
                f"recall-{label}": report["recall"],
                f"f1-{label}": report["f1-score"],
            }
        )

    result["accuracy"] = test_report["accuracy"]
    result["macro f1-score"] = test_report["macro avg"]["f1-score"]

    try:
        df = pd.read_csv(OUTPUT_DIR + "results.csv")
        df = df.append(result, ignore_index=True)
        df.to_csv(OUTPUT_DIR + "results.csv", index=False)
    except FileNotFoundError:
        df = pd.DataFrame(result, index=[0])
        df.to_csv(OUTPUT_DIR + "results.csv", index=False)

    # save the confusion matrix of the model in png file
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(OUTPUT_DIR + "{}.png".format(experiment_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))


def main():

    # Obtain model/experiment name
    # based on the input setting
    experiment_name = set_args()

    output = pd.read_csv(f"{OUTPUT_DIR}{experiment_name}.csv")
    Y_test = output["Test"]
    Y_predict = output["Predict"]

    save_results(Y_test, Y_predict, f"{experiment_name}")


if __name__ == "__main__":
    main()
