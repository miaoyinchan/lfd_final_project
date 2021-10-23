import os
import pandas as pd

TEST_DIR = "train-test-dev/test.csv"
TRAIN_DIR = "train-test-dev/train.csv"
DEV_DIR = "train-test-dev/dev.csv"

def load_data():

    """ Return train, test and dev sets as dataframe """

    train = pd.read_csv(TRAIN_DIR)
    test = pd.read_csv(TEST_DIR)
    dev = pd.read_csv(DEV_DIR)
    return train, test, dev

def main():

    train, test, dev = load_data()
    
    #concate train, test, dev sets into single dataset
    dataset = pd.concat([train,test,dev], ignore_index=True, axis=0)
    print(dataset.shape)




if __name__ == "__main__":
    main()