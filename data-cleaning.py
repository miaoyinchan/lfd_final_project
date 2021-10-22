import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd


nltk.download("punkt")


TEST_DIR = "train-test-dev/test.csv"
TRAIN_DIR = "train-test-dev/train.csv"
DEV_DIR = "train-test-dev/dev.csv"


def load_data():
    train = pd.read_csv(TRAIN_DIR)
    test = pd.read_csv(TEST_DIR)
    dev = pd.read_csv(DEV_DIR)
    return train, test, dev


def clean_data(df):

    cleaned_articles = list()
    for article in df["article"]:
        article = article.replace("\n", " ")
        sen_list = sent_tokenize(article)

        # concate in list removing last two
        article = " ".join(sen_list[:-2])

        cleaned_articles.append(article)

    df["article"] = cleaned_articles

    return df


def main():

    train, test, dev = load_data()

    train_cleaned = clean_data(train)
    test_cleaned = clean_data(test)
    dev_cleaned = clean_data(dev)

    train_cleaned.to_csv(TRAIN_DIR)
    test_cleaned.to_csv(TEST_DIR)
    dev_cleaned.to_csv(DEV_DIR)


if __name__ == "__main__":
    main()
