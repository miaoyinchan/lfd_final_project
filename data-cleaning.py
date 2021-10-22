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


def clean_text(text):
    rgx_list = [
        # PARIS: or PARIS, or PARIS --
        r"^[A-Z]+ ?(:|--|,)",
        # Paris, Dec. 1 --
        r"^[A-Za-z]+, [A-Za-z]+\. [0-9]+ --",
        # New Delhi, Dec. 7 --
        r"^[[A-Za-z]+ [[A-Za-z]+, [A-Za-z]+. [0-9]+ --",
        # SAN DIEGO --
        r"^[[A-Za-z]+ [[A-Za-z]+ (:|--|,)",
        # NEW DELHI -
        r"^[[A-Za-z]+ [[A-Za-z]+ (:|-|,)",
        # LE BOURGET, France --
        r"^[[A-Za-z]+ [[A-Za-z]+(:|--|,) [[A-Za-z]+ (:|--|,)",
        #  England --
        r"^ [[A-Za-z]+ (:|--|,)",
        # WASHINGTON —
        r"^[[A-Za-z]+ (:|—|,)",
        # BEIJING -
        r"^[[A-Za-z]+ (:|-|,)",
        # NEW DELHI:
        r"^[[A-Za-z]+ [[A-Za-z]+(:|-|,)",
        # NEW DELHI/BEIJING:
        r"^[[A-Za-z]+ [[A-Za-z]+(:|-|,|\/)[[A-Za-z]+(:|-|,|\/)",
    ]
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, "", new_text, flags=re.MULTILINE)
    return new_text


def clean_data(df):
    cleaned_articles = list()
    for article in df["article"]:
        article = article.replace("\n", " ")
        sen_list = sent_tokenize(article)

        # concate in list removing last two
        article = " ".join(sen_list[:-2])
        article = clean_text(article)

        cleaned_articles.append(article)

    df["article"] = cleaned_articles

    df = df.dropna()

    return df


def main():
    train, test, dev = load_data()

    train_cleaned = clean_data(train)
    test_cleaned = clean_data(test)
    dev_cleaned = clean_data(dev)

    train_cleaned.to_csv(TRAIN_DIR, index=False)
    test_cleaned.to_csv(TEST_DIR, index=False)
    dev_cleaned.to_csv(DEV_DIR, index=False)


if __name__ == "__main__":
    main()
