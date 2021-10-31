
import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

TRAIN_DIR = "train-test-dev/"


def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(TRAIN_DIR+'train.csv')

    return train

def augment_text(train):

    texts = train[train['topic']=='CLIMATE']
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")

    new_data = []
    index = [i for i in range(texts.shape[0])]
    for i in tqdm(index):
        row = texts.iloc[i]
        new_row = {k:row[k] for k in row.keys() if k !='article'}
        new_row['article'] = aug.augment(row['article'])
        new_data.append(new_row)


    new_df = pd.DataFrame(new_data)
    return pd.concat([train,new_df], ignore_index=True, axis=0)


def main():

    train = load_data()
    train_aug = augment_text(train)
    train_aug.to_csv(TRAIN_DIR+'train_aug.csv', index= False)

    




if __name__ == "__main__":
    main()