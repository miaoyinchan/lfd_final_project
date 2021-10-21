import json
import os
from pathlib import Path
from collections import Counter


def main():
    documents, newspapers, labels = create_data("COP_filt3_sub")
    newspapers = [fix_newspaper_name(name) for name in newspapers]

    # print(f"Documents: {len(documents)}")
    print(f"Newspapers: {len(newspapers)}")
    print(f"Labels: {len(labels)}")

    print("")

    label_count = Counter(labels)
    for label, count in label_count.items():
        print(f"{label}: {count}")

    print("")

    newspaper_count = Counter(newspapers)
    for newspaper, count in newspaper_count.items():
        print(f"{newspaper}: {count}")


def create_data(path):
    files = Path(path)
    documents = []
    classifications = []
    newspapers = []

    for f in files.rglob("*.json"):
        for article in load_data(f):
            newspapers.append(article["newspaper"])
            documents.append(article["body"])

    labels = label_political_orientation(newspapers)

    return documents, newspapers, labels


def fix_newspaper_name(name):
    names = {
        "Sydney Morning Herald (Australia)": "Sydney Morning Herald",
        "The Age (Melbourne, Australia)": "The Age",
        "The Times of India (TOI)": "The Times of India",
        "The Times (South Africa)": "The Times",
        "Mail & Guardian": "Mail and Guardian",
        "The New York Times": "New York Times",
    }
    return names.get(name) or name


def label_political_orientation(newspapers):
    newspaper_to_label = {
        "The Australian": "Right-Center",
        "Sydney Morning Herald": "Left-Center",
        "Sydney Morning Herald (Australia)": "Left-Center",
        "The Age": "Left-Center",
        "The Age (Melbourne, Australia)": "Left-Center",
        "The Times of India": "Right-Center",
        "The Times of India (TOI)": "Right-Center",
        "The Hindu": "Left-Center",
        "The Times": "Right-Center",
        "The Times (South Africa)": "Right-Center",
        "Mail and Guardian": "Left-Center",
        "Mail & Guardian": "Left-Center",
        "The Washington Post": "Left-Center",
        "New York Times": "Left-Center",
        "The New York Times": "Left-Center",
    }


    labels = [
        newspaper_to_label[newspaper]
        for newspaper in newspapers
    ]

    return labels


def load_data(file):
    with file.open("r") as fd:
        data = json.load(fd)

    articles = []

    for article in data["articles"]:
        articles.append({
            "newspaper": article["newspaper"],
            "body": article["body"],
        })

    return articles


if __name__ == "__main__":
    main()
