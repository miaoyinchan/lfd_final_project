import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

TRAIN_DIR = "train-test-dev/"


def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(TRAIN_DIR+'train.csv')

    return train

def augment_text(data):

    """ Return new augmented data merged with previous training set"""

    texts = data[data['topic']=='CLIMATE']

    #initilize Augmenter class that searches for top n similar words using contextual word embeddings
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

    new_data = list()
    index = [i for i in range(texts.shape[0])]
    for i in tqdm(index):
        row = texts.iloc[i]
        new_row = {k:row[k] for k in row.keys() if k !='article'}
        #get new article by inserting contexualized similar words
        new_row['article'] = aug.augment(row['article'])
        new_data.append(new_row)


    new_df = pd.DataFrame(new_data)
    return pd.concat([data,new_df], ignore_index=True, axis=0)


def downsampling(data):

    """ Return balanced train set after removing MISC articles randomly"""

    misc = data[data['topic']=='MISC']
    others = data[data['topic']=='CLIMATE']

    n = data["topic"].value_counts().to_dict()["CLIMATE"]

    #select random n misc articles from train set 
    misc = misc.sample(n, random_state= 1)

    return pd.concat([misc,others], ignore_index=True, axis=0)


def main():

    #load train set into pandas dataframe
    train = load_data()
    
    #get augmented data
    train_aug = augment_text(train)
    
    #save augmented as csv file
    train_aug.to_csv(TRAIN_DIR+'train_aug.csv', index= False)

    #get balanced train set after random downlsampling
    train_down = downsampling(train_aug)

    #save data in csv format
    train_down.to_csv(TRAIN_DIR+'train_down.csv', index= False)
    




if __name__ == "__main__":
    main()