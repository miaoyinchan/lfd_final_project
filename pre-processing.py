import json
import os
import pandas as pd

DIR = "data/"


GROUP = {'CLIMATE': ["CLIMATE CHANGE", "CLIMATOLOGY", "CLIMATE CHANGE REGULATION & POLICY", "WEATHER", "GLOBAL WARMING'"], 
        "EMISSIONS": ['EMISSIONS', 'GREENHOUSE GASES', 'POLLUTION & ENVIRONMENTAL IMPACTS', 'AIR QUALITY REGULATION', 'AIR POLLUTION'],
        }


def match_list(subjects, group):
    
    percentage = list()
    
    for subject in subjects:
        name = subject['name']
        p = subject['percentage']
        if name in group:
            if p != '':
                percentage.append(int(p))
                

    if len(percentage)==0:
        return 0
    
   
    return max(percentage)
            
             

def load_data(dir):
    
    files = os.listdir(dir)
    dataset = list()
    for f in files:
        with open(dir+f,) as file:
            file = json.load(file)
            for article in file['articles']:

                data = dict()
                data['cop_edition'] = file['cop_edition']
                data['newspaper'] = article['newspaper']
                data['headline'] = article['headline']
                data['date'] = article['date']
                data['article'] = article['body']

                subjects = article['classification']['subject']
                if subjects is None:
                    continue

                data['subjects'] = subjects

                dataset.append(data)
    
    return dataset      


def get_label(subjects):
    
    match = {label:match_list(subjects,topics) for label,topics in GROUP.items() if match_list(subjects,topics)!=0}  

    topics = list(match.keys())

    if len(topics)==1 and match[topics[0]]>=75.00:            
        return topics[0]

    elif len(topics)==0:
            return 'MISC'
    else:
        return None
    
     

def label_data(dataset):


    Labeled_dataset = list()
    for data in dataset:

        subjects = data['subjects']
        label = get_label(subjects)
        
        if label is not None:
            data['topic'] = label
            del data['subjects']
            Labeled_dataset.append(data)
    

    return Labeled_dataset

def select_random_rows(df,n,filter="MISC"):

    rdf = df[df['topic']==filter]
    rdf = rdf.groupby('cop_edition').sample(n=n, random_state=1)

    df = df[df['topic']!=filter]
    df = df.append(rdf, ignore_index=True)

    return df


def split_data(dataset):

    df = pd.DataFrame(dataset)
    Range_train = [str(i) for i in range(1,21)]
    Range_test = ["23","24"]
    Range_dev = ["21","22"]

    train = df.loc[df['cop_edition'].isin(Range_train)]
    test = df.loc[df['cop_edition'].isin(Range_test)]
    dev = df.loc[df['cop_edition'].isin(Range_dev)]

    train = select_random_rows(train,125)
    test = select_random_rows(test,300)
    dev = select_random_rows(dev,300)

    return train, dev, test
  


def print_count(df, name):
    
   print("{} \n {} \n".format(name, df['topic'].value_counts()))

def main():

    Raw_data = load_data(DIR)
    Labeled_data = label_data(Raw_data)
        
    train, dev, test = split_data(Labeled_data)

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    dev.to_csv('dev.csv', index=False)

    print_count(train,'train')
    print_count(test, 'test')
    print_count(dev, 'dev')


if __name__ == "__main__":
    main()

