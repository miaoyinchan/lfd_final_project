import logging
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import joblib
import utils


def load_data(dir, training_set):

    """Return appropriate training and validation sets reading from csv files"""

    if training_set.upper()=="RESAMPLE":
        df = pd.read_csv(dir+'/train_aug.csv')
    elif training_set.upper()=="RESAMPLE-BALANCE":
        df = pd.read_csv(dir+'/train_down.csv')
    elif training_set.upper()=="FULL":
        df = pd.read_csv(dir+'/train.csv')

    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    #get parameters for experiments
    config, model_name = utils.get_config()

    #get n-gram parameters from config
    
    n1 = config['n1'] #lower end of the word n-gram range
    n2 = config['n2'] #upper end of the word n-gram range

    #initialize vectorizer
    if config['vectorizer'].upper()=="TF-IDF":
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))
    else:
        vec = CountVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))


    #Create Log file
    try:
        os.mkdir(utils.LOG_DIR)
        log = logging.basicConfig(filename=utils.LOG_DIR+model_name+'.log',level=logging.INFO)
        print = log
    except OSError as error:
        logging.basicConfig(filename=utils.LOG_DIR+model_name+'.log', level=logging.INFO)
        log = logging.basicConfig(filename=utils.LOG_DIR+model_name+'.log',level=logging.INFO)
        print = log
    

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    #load data from train-test-dev folder
    X_train, Y_train = load_data(utils.DATA_DIR, config['training-set'])

    # Train the model with training set
    classifier.fit(X_train, Y_train)

    #save parameter in log
    logging.info(classifier.get_params())

    #save model output directory
    try:
        os.mkdir(utils.MODEL_DIR)
        joblib.dump(classifier, utils.MODEL_DIR+model_name, compress=9)
        
    except OSError as error:
        joblib.dump(classifier, utils.MODEL_DIR+model_name, compress=9)
    

if __name__ == "__main__":
    main()
