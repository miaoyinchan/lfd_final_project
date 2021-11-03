import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np

#set plot parameter (font size)
plt.rcParams.update({'font.size': 10})



def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv("train-test-dev/train.csv")
    test = pd.read_csv("train-test-dev/test.csv")
    dev = pd.read_csv("train-test-dev/dev.csv")
    train_resample = pd.read_csv("train-test-dev/train_aug.csv")
    train_resample_balance = pd.read_csv("train-test-dev/train_aug.csv")
    return train, test, dev, train_resample, train_resample_balance

def find_distribution(df, column_name):

    """Return the frequency of each class in different newspaper or COP meetings as a dictionary"""

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



def plot_distribution(dataset, column_name):

    """Plot distributions of classes per meeting"""

    dist = find_distribution(dataset, column_name)
    labels = [str(x) for x in list(dist.keys())]
    climate = [freq['CLIMATE'] for freq in dist.values()]

    width = 0.35      

    fig, ax = plt.subplots(figsize=(22,8))

    ax.bar(labels, climate, width, label='CLIMATE')
    ax.set_xlabel(column_name.replace("_"," "))
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of topics per '+column_name.replace("_"," "))
    
    ax.legend()
    plt.savefig("Figures/"+column_name.replace("_"," ") + "-distributions.jpg")


def token_count(dataset):

    """Return the number average tokens in dataset along with number of articles greater than 512 tokens"""

    cnt = 0
    larger_articles = 0
    for data in dataset:
        token_length = len(word_tokenize(data))
        cnt+= token_length
        if token_length > 512:
            larger_articles +=1

        

    return cnt/dataset.shape[0], larger_articles



def plot_pie(distribution):

    """Plot a pie chart from class distribution"""

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

    """Convert a list to a dictionary to count the number of times an item appears and return the sorted dictionary"""

    frequency = {}
    for t in topics:
        if t in frequency.keys():
            frequency[t]+=1
        else:
            frequency[t] =1

    #sort the frequency dictionary in descending order
    frequency = {k:v for k,v in sorted(frequency.items(), key= lambda item:item[1], reverse=True)}
    return frequency



def topics_with_max_pct(data):

  """Return a list of topics with the highest percentages"""

  #get the maximum percentage
  mx = max(data.values())
  return [k for k,v in data.items() if v==mx]  




def data_summary(dataset):

    """Print the summary of data e.g total number of articles, number of unique subjects"""
    
    number_articles = [data['article'] for data in dataset if data['article'] is not None]
    print("The number of articles {}".format(len(number_articles)))

    #seperate the subjects from the dataset
    dataset = [data['subjects'] for data in dataset if data['subjects'] is not None]
    
    #count and print the number unique subjects 
    topics = []
    for idx, item in enumerate(dataset):
        try:
            names = [value['name'] for value in item]
            topics.extend(names)
        except:
            continue

    topics = count_frequency(topics)
    number_topics = len(topics.keys())
    print("\nThe number of unique subjects {}".format(number_topics))
    
    #count and print the number unique subjects with maximum percentage 
    mx_topics = []
    for idx, item in enumerate(dataset):
        try:
            dct = {value['name']: int(value['percentage']) for value in item if value['percentage'] !=""}
        except:
            continue
        
        if len(dct)!=0:
            mx_topics.extend(topics_with_max_pct(dct))

    mx_topics = count_frequency(mx_topics)
    number_mx_topics = len(mx_topics.keys())
    print("\nThe number of unique subjects with maximum percentage {}".format(number_mx_topics))



def main():

    train, test, dev, train_resample, train_resample_balance = load_data()
    
    #concate train, test, dev sets into single dataset
    dataset = pd.concat([train,test,dev], ignore_index=True, axis=0)

    
    #print average token counts
    print("Average Token CLIMATE: {:.2f}".format(token_count(dataset[dataset['topic']=="CLIMATE"]['article'])[0]))
    print("Average Token MISC: {:.2f}".format(token_count(dataset[dataset['topic']=="MISC"]['article'])[0]))
   
    #print distribution counts
    print("\n{} \n{} \n".format("train", train["topic"].value_counts()))
    print("{} \n{} \n".format("test", test["topic"].value_counts()))
    print("{} \n{} \n".format("dev", dev["topic"].value_counts()))
    print("{} \n{} \n".format("train upsampling", train_resample["topic"].value_counts()))
    print("{} \n{} \n".format("train balance", train_resample_balance["topic"].value_counts()))
   
    dist_full = dataset['topic'].value_counts()
    print("{} \n{} \n".format("Full Distribution", dist_full))

    #plot distribution using a bar chart and pie chart
    try:
        #create directory for figures
        os.mkdir("Figures")
        plot_distribution(dataset, "cop_edition")
        plot_distribution(dataset, "newspaper")
        plot_pie(dist_full.to_dict())

    except OSError as error:
        plot_distribution(dataset, "cop_edition")
        plot_distribution(dataset, "newspaper")
        plot_pie(dist_full.to_dict())

    

    #print the summary of data
    dir = "data/"
    dataset = load_full_data(dir)
    data_summary(dataset)





if __name__ == "__main__":
    main()