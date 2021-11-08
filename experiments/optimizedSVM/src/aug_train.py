#!/usr/bin/env python

"""Training Linear SVM model using
   augmentation data and a combination
   of up- and downsampling. Then tuning
   to find the hyper-parameter (only
   for the model using augmentation
   data because experiment showed
   that it has a better performance)."""


import argparse
import csv
import logging
import sys

import joblib
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import (
    saveModel,
    read_data,
    train_model,
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
    Return
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ud",
        "--updownsampling",
        action="store_true",
        help="Upsampling on minority class and downsampling "
             "on majority class",
    )
    parser.add_argument(
        "-b",
        "--bestmodel",
        action="store_true",
        help="Train model only with the hyper-parameter c=1",
    )

    args = parser.parse_args()
    return args


def main():
    args = create_arg_parser()

    # Create Log file
    logging.basicConfig(filename=f"{LOG_DIR}/optSVM.log", level=logging.INFO)


    # Load training set after up- and downsampling then use it
    # to train and save model.
    if args.updownsampling:
        X_train, Y_train = read_data(f"{DATA_DIR}/train_down.csv")

        train_model(1, X_train, Y_train, "model_updownsampling_aug")

    # Load training set after upsampling then use it
    # to train and save model.
    else:
        X_train, Y_train = read_data(f"{DATA_DIR}/train_aug.csv")

        # Select this option, only model with hyperparameter c=1 trained
        if args.bestmodel:
            train_model(1, X_train, Y_train, "model_upsampling_aug_1")

        else:
            # Tune hyperparameters in this range of c value for
            # model trained with augmentation data
            C_values = [1, 10, 100, 1000]
            for c in C_values:
                train_model(c, X_train, Y_train, f"model_upsampling_aug_{c}")



if __name__ == "__main__":
    main()

