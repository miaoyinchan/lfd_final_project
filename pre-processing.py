import json
import os
import pandas as pd

DIR = "data/"


GROUP = {
    "CLIMATE": [
        "CLIMATE CHANGE",
        "CLIMATOLOGY",
        "CLIMATE CHANGE REGULATION & POLICY",
        "WEATHER",
        "GLOBAL WARMING'",
    ],
    "EMISSIONS": [
        "EMISSIONS",
        "GREENHOUSE GASES",
        "POLLUTION & ENVIRONMENTAL IMPACTS",
        "AIR QUALITY REGULATION",
        "AIR POLLUTION",
    ],
}


def match_list(subjects, group):

    """
    Compare an article's subjects to a pre-defined set of subjects from the CLIMATE or EMISSIONS groups,
    and Return the highest percentage from the subjects that is matched.

    """

    # get the percentages of subjects that is present on both article and group
    percentage = [
        int(subject["percentage"])
        for subject in subjects
        if subject["name"] in group and subject["percentage"] != ""
    ]

    # If there are no matches, return 0
    if len(percentage) == 0:
        return 0

    return max(percentage)


def load_data(dir):

    """
    read json files from data folder,
    extract cop edition, newspaper name, headline, published date, article body, subjects from each file
    and return the dataset
    """

    files = os.listdir(dir)
    dataset = list()
    for f in files:
        with open(
            dir + f,
        ) as file:
            file = json.load(file)
            for article in file["articles"]:

                data = dict()
                data["cop_edition"] = file["cop_edition"]

                if data["cop_edition"] == '6a':
                    data["cop_edition"] = '6'

                data["newspaper"] = article["newspaper"]
                data["headline"] = article["headline"]
                data["date"] = article["date"]
                data["article"] = article["body"]

                subjects = article["classification"]["subject"]
                if subjects is None:
                    continue

                data["subjects"] = subjects

                dataset.append(data)

    return dataset


def get_label(subjects):

    """
    Return the name of that group as label if the article is matched with only one group and the maximum percentage is more than or equal to 75.
    If the article is matched with more than one group, return None. If no group matches, use MISC as the label.

    """

    # obtain the names of the groups that correspond to the article's topic, as well as the maximum percentage
    match = {
        label: match_list(subjects, topics)
        for label, topics in GROUP.items()
        if match_list(subjects, topics) != 0
    }

    topics = list(match.keys())

    if len(topics) == 1 and match[topics[0]] >= 75.00:
        return topics[0]

    elif len(topics) == 0:
        return "MISC"
    else:
        return None


def label_data(dataset):

    """If each article meets the criteria, add a label to it and returns the tagged dataset"""

    Labeled_dataset = list()
    for data in dataset:

        subjects = data["subjects"]

        label = get_label(subjects)

        if label is not None:
            data["topic"] = label
            del data["subjects"]
            Labeled_dataset.append(data)

    return Labeled_dataset


def select_random_rows(df, n, filter="MISC"):

    """Return a dataframe comprising only n MISC articles at random"""

    rdf = df[df["topic"] == filter]
    rdf = rdf.groupby("cop_edition").sample(n=n, random_state=1)

    df = df[df["topic"] != filter]
    df = df.append(rdf, ignore_index=True)

    return df


def split_data(dataset):

    """Split the dataset into three sections: training, validation, and testing.
    The most recent meeting is being utilized for testing purposes.
    The meeting before to the last one is set aside for validation, while the others are utilized for training.
    """

    df = pd.DataFrame(dataset)
    meetings = df['cop_edition'].unique()
    meetings = sorted([m for m in meetings], key=lambda x:int(x))
    Range_train = meetings[:-2]
    Range_test = [meetings[-1]]
    Range_dev = [meetings[-2]]

    train = df.loc[df["cop_edition"].isin(Range_train)]
    test = df.loc[df["cop_edition"].isin(Range_test)]
    dev = df.loc[df["cop_edition"].isin(Range_dev)]

    # For training, 125 MISC articles are randomly selected from each meeting except the last two
    train = select_random_rows(train, 125)

    # For test, 250 MISC articles are randomly selected from the latest meeting
    test = select_random_rows(test, 250)

    # For dev, 250 MISC articles are randomly selected from the meeting before to the last one
    dev = select_random_rows(dev, 250)

    return train, dev, test


def main():

    # load raw json files
    Raw_data = load_data(DIR)

    # tag articles with labels
    Labeled_data = label_data(Raw_data)

    # split the data into three sets
    train, dev, test = split_data(Labeled_data)

    # Save training, development, and testing sets in csv format
    try:
        #create directory for train-test-dev sets
        directory = "train-test-dev"
        os.mkdir(directory)

    except OSError as error:
         directory = "train-test-dev"

    train.to_csv(directory+"/train.csv", index=False)
    test.to_csv(directory+"/test.csv", index=False)
    dev.to_csv(directory+"/dev.csv", index=False)

    # Print the group distribution in train, dev, and test sets
    print("{} \n{} \n".format("train", train["topic"].value_counts()))
    print("{} \n{} \n".format("test", test["topic"].value_counts()))
    print("{} \n{} \n".format("dev", dev["topic"].value_counts()))


if __name__ == "__main__":
    main()
