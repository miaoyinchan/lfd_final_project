import json
import os
import pandas as pd

dir = "data/"
files = os.listdir(dir)


def matchList(l1,l2):

    for l in l1:
        if l in l2:
            return True



group = {'CLIMATE': ["CLIMATE CHANGE", "CLIMATOLOGY", "CLIMATE CHANGE REGULATION & POLICY", "WEATHER"], "EMISSIONS": ['EMISSIONS', 'GREENHOUSE GASES', 'POLLUTION & ENVIRONMENTAL IMPACTS', 'AIR QUALITY REGULATION', 'AIR POLLUTION']
, "GLOBAL WARMING": ['GLOBAL WARMING']}

freq = {'CLIMATE': 0,'EMISSIONS': 0,"GLOBAL WARMING": 0, 'MISC': 0}

dataset = list()

for f in files:
    file = open(dir+f,)
    file = json.load(file)
    for article in file['articles']:


        subjects = article['classification']['subject']
        if subjects is None:
            continue

        names = [d['name']  for d in subjects]
        match = [k for k,v in group.items() if matchList(names,v)]


        if len(match)==1:
            freq[match[0]]+=1
            topic = match[0]
        elif len(match)==0:
            freq['MISC']+=1
            topic = 'MISC'
        else:
            continue

        data = dict()
        data['cop_edition'] = file['cop_edition']
        data['newspaper'] = article['newspaper']
        data['headline'] = article['headline']
        data['date'] = article['date']
        data['article'] = article['body']
        data['topic'] = topic

        dataset.append(data)


df = pd.DataFrame(dataset)
df.to_csv('topic-classification.csv',index=False)

print(len(dataset))

