import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


GROUPS = {
    "CLIMATE": [
        "CLIMATE CHANGE",
        "CLIMATOLOGY",
        "CLIMATE CHANGE REGULATION & POLICY",
        "WEATHER",
    ],
    "EMISSIONS": [
        "EMISSIONS",
        "GREENHOUSE GASES",
        "POLLUTION & ENVIRONMENTAL IMPACTS",
        "AIR QUALITY REGULATION",
        "AIR POLLUTION",
    ],
    "GLOBAL WARMING": [
        "GLOBAL WARMING",
    ],
}


def main():
    doc, labels = create_data("data")
    for label_name, label_count in Counter(labels).items():
        print(f"{label_name}: {label_count}")
    # test_label()


def create_data(data_directory):
    data_directory = Path(data_directory)
    labels = []

    for f in data_directory.rglob("*.json"):
        data = load_data(f)
        for article in data["articles"]:
            label = find_label_for_subjects(article["classification"]["subject"])
            if label is None:
                continue
            labels.append(label)

    return article, labels

    for label_name, label_count in Counter(labels).items():
        print(f"{label_name}: {label_count}")


def find_label_for_subjects(subjects):
    '''Find label for articles based on this logic,
    collect the articles' subjects of which percentage > 75%:
    - if there is no subject, get MISC label,
    - if all subjects belong to no group, get MISC label,
    - if all subjects belong to 1 group, get label,
    - if all subjects belong to >1 group, get no label,
    - if 1 of the subjects belong to 1 group, get label.
    '''

    # if there is no subject, get MISC label
    if subjects is None or len(subjects) == 0:
        return "MISC"

    #collect the articles' subjects of which percentage > 75%
    subjects_above_75pct = []
    for subject in subjects:
        try:
            pct = int(subject["percentage"])
        except ValueError:
            continue
        if pct > 75:
            subjects_above_75pct.append(subject["name"])

    # check if the subjects (>75pct) belongs to groups
    labels = set()
    for topic_name, subject_name in GROUPS.items():
        for subject in subjects_above_75pct:
            if subject in subject_name:
                labels.add(topic_name)

    # if all subjects belong to no group, get MISC label
    if len(labels) == 0:
        label = "MISC"

    # if all subjects belong to 1 group, get label
    # or if 1 of the subjects belong to 1 group, get label
    # (other subjects not belong to groups won't
    # include in the set)
    elif len(labels) == 1:
        label = list(labels)[0]

    # if all subjects belong to >1 group, get no label
    elif len(labels) > 1:
        label = None

    return label


def test_label():
    subjects = [
          {
            "name": "EMERGING MARKETS",
            "percentage": "90"
          },
          {
            "name": "ENVIRONMENT & NATURAL RESOURCES",
            "percentage": "90"
          },
          {
            "name": "EMISSIONS",
            "percentage": "89"
          }
    ]

    print(find_label_for_subjects(subjects))


def load_data(file):
    with file.open("r") as fd:
        return json.load(fd)


if __name__ == "__main__":
    main()
