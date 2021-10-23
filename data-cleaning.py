import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np

nltk.download("punkt")


TEST_DIR = "train-test-dev/test.csv"
TRAIN_DIR = "train-test-dev/train.csv"
DEV_DIR = "train-test-dev/dev.csv"


def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(TRAIN_DIR)
    test = pd.read_csv(TEST_DIR)
    dev = pd.read_csv(DEV_DIR)
    return train, test, dev


def clean_text(text):

    """ Return the cleaned data after removing special charachteres, web addresses, and datelines"""

    rgx_list = [
        # PARIS: or PARIS, or PARIS --
        r"^[A-Z]+ ?(:|--|,)",
        # Paris, Dec. 1 --
        r"^[A-Za-z]+, [A-Za-z]+\. [0-9]+ --",
        # New Delhi, Dec. 7 --
        r"^[A-Za-z]+ [A-Za-z]+, [A-Za-z]+. [0-9]+ --",
        # SAN DIEGO --
        r"^[A-Za-z]+ [A-Za-z]+ (:|--|,)",
        # NEW DELHI -
        r"^[A-Za-z]+ [A-Za-z]+ (:|-|,)",
        # LE BOURGET, France --
        r"^[A-Za-z]+ [A-Za-z]+(:|--|,) [A-Za-z]+ (:|--|,)",
        # 'WASHINGTON —', 'BEIJING -', ' England --'
        r"^[ ]?[A-Za-z]+ (:|—|-|--,)",
        # NEW DELHI:
        r"^[A-Za-z]+ [A-Za-z]+(:|-|,)",
        # NEW DELHI/BEIJING:
        r"^[A-Za-z]+ [A-Za-z]+(:|-|,|\/)[A-Za-z]+(:|-|,|\/)",
        # www.wildzoofari.org
        r"(?i)\b((www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    ]
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, "", new_text, flags=re.MULTILINE)
        #removing the special charachters such as â€™
        new_text = re.sub(r'[^a-zA-Z0-9 !,:;.?_-`"~&(){}*+\'_|]', "", new_text, flags=re.MULTILINE)
        
    return new_text


def clean_data(df):

    """ Return the cleaned dataset after removing extra lines and charachters """

    cleaned_articles = list()
    for article in df["article"]:
        article = article.replace("\n", " ")
        sen_list = sent_tokenize(article)

        # concate in list removing last two
        article = " ".join(sen_list[:-2])
        article = clean_text(article)

        cleaned_articles.append(article or np.nan)

    df["article"] = cleaned_articles

    return df


def main():
    
    train, test, dev = load_data()

    train_cleaned = clean_data(train)
    test_cleaned = clean_data(test)
    dev_cleaned = clean_data(dev)

    #remove rows with NaN values
    train_cleaned = train_cleaned.dropna()
    test_cleaned = test_cleaned.dropna()
    dev_cleaned = dev_cleaned.dropna()

    train_cleaned.to_csv(TRAIN_DIR, index=False)
    test_cleaned.to_csv(TEST_DIR, index=False)
    dev_cleaned.to_csv(DEV_DIR, index=False)


if __name__ == "__main__":
    main()
