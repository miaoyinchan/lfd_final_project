import os
from nltk.util import pr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({'font.size': 10})

TEST_DIR = "train-test-dev/test.csv"
TRAIN_DIR = "train-test-dev/train.csv"
DEV_DIR = "train-test-dev/dev.csv"

def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(TRAIN_DIR)
    test = pd.read_csv(TEST_DIR)
    dev = pd.read_csv(DEV_DIR)
    return train, test, dev

def find_distribution(df,column_name):

    dist_count = df.groupby(column_name)['topic'].value_counts().to_dict()

    dist = {}
    for k,v in dist_count.items():
        
        col = k[0]
        topic = k[1]
        cnt = v

        if col not in dist.keys():
            dist[col] ={topic:cnt}
        else:
            dist[col][topic] =cnt

    return dist

    
    
def generate_word_cloud(freqs, title):

    """word cloud genertaed from frquency"""
    plt.figure(figsize=(30, 20))
    w = WordCloud(
        width=3000, height=2400, mode="RGBA", background_color="white", max_words=1000
    ).fit_words(freqs)
    plt.imshow(w)
    plt.axis("off")
    plt.savefig("Figures/"+title + "-word-cloud.jpg")


def plot_distribution(dataset, column_name):

    dist = find_distribution(dataset,column_name)
    labels = [str(x) for x in list(dist.keys())]
    climate = [freq['CLIMATE'] for freq in dist.values()]
    # emissions = [freq['EMISSIONS'] for freq in dist.values()]

    width = 0.35      

    fig, ax = plt.subplots(figsize=(22,8))

    ax.bar(labels, climate, width, label='CLIMATE')
    # ax.bar(labels, emissions, width, bottom=climate, label='EMISSIONS')
    ax.set_xlabel(column_name.replace("_"," "))
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of topics per '+column_name.replace("_"," "))
    
    ax.legend()
    plt.savefig("Figures/"+column_name.replace("_"," ") + "-distributions.jpg")


def token_count(dataset):

    cnt = 0
    larger_articles = 0
    for data in dataset:
        token_length = len(word_tokenize(data))
        cnt+= token_length
        if token_length > 512:
            larger_articles +=1

        

    return cnt/dataset.shape[0], larger_articles



def plot_pie(distribution):

    labels = list(distribution.keys())
    sizes = list(distribution.values())
    explode = (0, 0)  

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  

    plt.savefig("Figures/pie-distributions.png")

def load_full_data(dir):

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
                    data["subjects"] = np.nan
                else:
                    data["subjects"] = subjects

                industry = article["classification"]["industry"]
                if industry is None:
                    data["industry"] = np.nan
                else:
                    data["industry"] =industry

                org = article["classification"]["organization"]    
                if org is None:
                    data["organization"] = np.nan
                else:
                    data["organization"] = org

               

                dataset.append(data)

    return dataset

def count_frequency(topics):
    
    frequency = {}
    for t in topics:
        if t in frequency.keys():
            frequency[t]+=1
        else:
            frequency[t] =1

    frequency = {k:v for k,v in sorted(frequency.items(), key= lambda item:item[1], reverse=True)}
    return frequency

def topics_with_max_pct(data):


  mx = max(data.values())
  return [k for k,v in data.items() if v==mx]  




def data_summary(dataset):

    
    number_articles = [data['article'] for data in dataset if data['article'] is not None]
    print("The number of articles {}".format(len(number_articles)))

    dataset = [data['subjects'] for data in dataset if data['subjects'] is not None]
    
    topics = []
    for idx, item in enumerate(dataset):
        try:
            names = [value['name'] for value in item]
            topics.extend(names)
        except:
            continue

    
    mx_topics = []
    for idx, item in enumerate(dataset):
        try:
            dct = {value['name']: int(value['percentage']) for value in item if value['percentage'] !=""}
        except:
            continue
        
        if len(dct)!=0:
            mx_topics.extend(topics_with_max_pct(dct))
        
 
    topics = count_frequency(topics)
    number_topics = len(topics.keys())
    print("The number of unique subjects {}".format(number_topics))


    mx_topics = count_frequency(mx_topics)
    number_mx_topics = len(mx_topics.keys())
    print("The number of unique subjects with maximum percentage {}".format(number_mx_topics))

    




def main():

    train, test, dev = load_data()
    
    #concate train, test, dev sets into single dataset
    dataset = pd.concat([train,test,dev], ignore_index=True, axis=0)

    

    print("Average Token CLIMATE: {}".format(token_count(dataset[dataset['topic']=="CLIMATE"]['article'])))
    print("Average Token MISC: {}".format(token_count(dataset[dataset['topic']=="MISC"]['article'])))
   
    print(train['topic'].value_counts().to_dict())
    print(test['topic'].value_counts().to_dict())
    print(dev['topic'].value_counts().to_dict())

    dist_full = dataset['topic'].value_counts().to_dict()

    print(dist_full)


    plot_distribution(dataset, "cop_edition")
    plot_distribution(dataset, "newspaper")

    plot_pie(dist_full)

    dir = "data/"
    dataset = load_full_data(dir)
    data_summary(dataset)





if __name__ == "__main__":
    main()