import re
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np


DIR = "train-test-dev/"


def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(DIR+"train.csv")
    test = pd.read_csv(DIR+"test.csv")
    dev = pd.read_csv(DIR+"dev.csv")
    
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


def split_train_data(df, n=5, labels = ['MISC','CLIMATE']):

    """Return a dataframe comprising only n MISC articles at random"""

    df_misc = df[df["topic"] == labels[0]]
    
    df_misc = df_misc.sample(n=int(df_misc.shape[0]/n), random_state=1)

    df_climate = df[df["topic"] == labels[1]]
    df_climate =df_climate.sample(n=int(df_climate.shape[0]/n), random_state=1)

    df_small = df_misc.append(df_climate, ignore_index=True)

    return df_small


def main():
    
    train, test, dev = load_data()

    #clean data
    train_cleaned = clean_data(train)
    test_cleaned = clean_data(test)
    dev_cleaned = clean_data(dev)

    #remove rows with NaN values
    train_cleaned = train_cleaned.dropna()
    test_cleaned = test_cleaned.dropna()
    dev_cleaned = dev_cleaned.dropna()

    #split train data for paramater optimization
    train_opt = split_train_data(train_cleaned)

    #save dataframes as csv files
    train_cleaned.to_csv(DIR+"train.csv", index=False)
    test_cleaned.to_csv(DIR+"test.csv", index=False)
    dev_cleaned.to_csv(DIR+"dev.csv", index=False)
    train_opt.to_csv(DIR+"train_opt.csv", index=False)


if __name__ == "__main__":
    main()
