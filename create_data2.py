import json
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
    # create_data("data")
    test_label()


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


    for label_name, label_count in Counter(labels).items():
        print(f"{label_name}: {label_count}")


def find_label_for_subjects(subjects):
    '''Based on the percentage of the subjects, label the article:
    - if only 1 highest percentage, if subject is one of the groups,
    get label, if not in the groups, get MISC.
    - if more than 1 highest percentage:
        + if all in the same group, get label.
        + if not in the same group, skip the sample.
        + if in no group, MISC.
    '''

    subjects_by_pct = {}

    if subjects is None or len(subjects) == 0:
        return "MISC"

    for subject in subjects:
        try:
            pct = int(subject["percentage"])
        except ValueError:
            continue
        if pct not in subjects_by_pct:
            subjects_by_pct[pct] = []
        subjects_by_pct[pct].append(subject["name"])

    if len(subjects_by_pct) == 0:
        return "MISC"

    highest_pct = max(subjects_by_pct.keys())
    highest_subjects = subjects_by_pct[highest_pct]

    labels = set()
    for label_name, label_subjects in GROUPS.items():
        for subject in highest_subjects:
            if subject in label_subjects:
                labels.add(label_name)

    if len(labels) == 0:
        label = "MISC"
    elif len(labels) == 1:
        label = list(labels)[0]
    elif len(labels) > 1:
        label = None

    return label


def test_label():
    subjects = [
        {
            "name": "EMERGING MARKETS",
            "percentage": "90",
        },
        {
            "name": "ENVIRONMENT & NATURAL RESOURCES",
            "percentage": "90",
        },
        {
            "name": "EMISSIONS",
            "percentage": "89",
        },
    ]

    print(find_label_for_subjects(subjects))


def load_data(file):
    with file.open("r") as fd:
        return json.load(fd)


if __name__ == "__main__":
    main()
