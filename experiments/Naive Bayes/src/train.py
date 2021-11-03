import logging
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import joblib
import json

DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
LOG_DIR = "../Logs/"



def get_config():

    """Return model name and paramters after reading it from json file"""
    try:
        location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v).upper() for v in configs.values()]
            model_name = "_".join(vals)
        return configs, model_name
    except FileNotFoundError as error:
        print(error)


def load_data(dir, experiment):

    """Return appropriate training and validation sets reading from csv files"""

    if experiment.upper()=="RESAMPLE":
        df = pd.read_csv(dir+'/train_aug.csv')
    elif experiment.upper()=="RESAMPLE-BALANCE":
        df = pd.read_csv(dir+'/train_down.csv')
    elif experiment.upper()=="FULL":
        df = pd.read_csv(dir+'/train.csv')

    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    #get parameters for experiments
    config, experiment_name = get_config()

    #get n-gram parameters from config
    n1 = config['n1']
    n2 = config['n2']

    #initialize vectorizer
    if config['vector'].upper()=="TF-IDF":
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))
    elif config['vector'].upper()=="CV":
        vec = CountVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))


    #Create Log file
    try:
        os.mkdir(LOG_DIR)
        log = logging.basicConfig(filename=LOG_DIR+experiment_name+'.log',level=logging.INFO)
        print = log
    except OSError as error:
        logging.basicConfig(filename=LOG_DIR+experiment_name+'.log', level=logging.INFO)
        log = logging.basicConfig(filename=LOG_DIR+experiment_name+'.log',level=logging.INFO)
        print = log
    

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    #load data from train-test-dev folder
    X_train, Y_train = load_data(DATA_DIR, config['experiment'])

    # Train the model with training set
    classifier.fit(X_train, Y_train)

    #save parameter in log
    logging.info(classifier.get_params())

    #save model output directory
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
        
    except OSError as error:
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
    

if __name__ == "__main__":
    main()