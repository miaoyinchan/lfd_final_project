import logging
import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import joblib


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
LOG_DIR = "../Logs/"

def create_arg_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")

    parser.add_argument("-n1", "--n1", default=1, type=int,
                        help="Ngram Start point")
    
    parser.add_argument("-n2", "--n2", default=1, type=int,
                        help="Ngram End point")


    args = parser.parse_args()
    return args

def load_data(dir):

    df = pd.read_csv(dir+'/train.csv')
    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    
    args = create_arg_parser()
    n1 = args.n1
    n2 = args.n2

    if args.tfidf:
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))
    else:
        vec = CountVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))

    if args.tfidf:
        experiment_name = "NB+Tf-idf+"+str(n1)+"-"+str(n2)
    else:
        experiment_name = "NB+CV+"+str(n1)+"-"+str(n2)

    #Create Log file
    try:
        os.mkdir(LOG_DIR)
        logging.basicConfig(filename=LOG_DIR+experiment_name+'.log',level=logging.INFO)
    except OSError as error:
        logging.basicConfig(filename=LOG_DIR+experiment_name+'.log', level=logging.INFO)
    

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    #load data from train-test-dev folder
    X_train, Y_train = load_data(DATA_DIR)

    # Train the model with training set
    classifier.fit(X_train, Y_train)

    #save parameter in log
    logging.info(classifier.get_params())

    #save model
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
        
    except OSError as error:
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
    

if __name__ == "__main__":
    main()
