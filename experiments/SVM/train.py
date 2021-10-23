#!/usr/bin/env python
import pandas as pd
import os
import csv
import argparse
import random
import nltk
import logging
from nltk.tokenize import word_tokenize
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import numpy as np  
from nltk.util import ngrams, pr
import sys
import joblib

DATA_DIR = '../../train-test-dev/'
MODEL_DIR = "Saved_Models/"
LOG_DIR = "Logs/"

def create_arg_parser():
    """
    Description:
    
    This method is an arg parser
    
    Return
    
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kernel",type=str, default='rbf',
                        help="Input kernel")
    parser.add_argument("-n1", "--n1", default=1, type=int,
                        help="Ngram Start point")
    parser.add_argument("-n2", "--n2", default=1, type=int,
                        help="Ngram End point")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    args = parser.parse_args()
    return args

def read_data(dataset):
    """
   """
    sentences = []
    labels = []
    file = open(dataset)
    text = list(csv.reader(file, delimiter=','))
    for row in text[1:]:
        tokens = row[-2].strip().split()
        sentences.append(" ".join(tokens))
        labels.append(row[-1])
    return sentences, labels

def saveModel(classifier,experiment_name ):
    #save model
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
        
    except OSError as error:
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)
    

def get_optimal_hyperParmeters(kernel, X_train, Y_train, X_test, Y_test, vec):
    """
    Description:
    
    This method vectorize the tokens and trains several SVM models
     using different hyperparameter values to find the best performing model
    
    Parameters:
    
    X_train = token lists for training
    
    Y_train = labels for training set
    
    X_test = token lists for testing/validating
    
    Y_test = labels for test/validation set
    
    """ 
     
    #For each model the C value increase by an order of magnitude
    List_C = list([0.0001,0.001,0.01,0.1,10,100,1000])
    C=0
    gamma = 0
    f1_opt = 0
    count = 0
    best_model = None
    logging.basicConfig(filename='logs/baseline_svm.log', encoding='utf-8', level=logging.DEBUG)
    if kernel == "rbf":
        #For each model the gamma value increase by an order of magnitude
        List_gamma = list([0.0001,0.001,0.01,0.1,10,100,1000])
        for i in List_C:
            progress(count, len(List_C),'C')
            count+=1
            for x in List_gamma:
                cls = SVC(kernel='rbf', C=i, gamma = x)
                classifier = Pipeline([('vec', vec), ('cls',cls)])
                classifier.fit( X_train, Y_train)
                pred = classifier.predict(X_test)
                f1 = report(Y_test, pred, digits=3, output_dict = True, zero_division= 0).get('macro avg').get('f1-score')
                if f1 > f1_opt:
                    best_model = classifier
                    f1_opt = f1
                    C = i
                    gamma = x
        logging.info(f"rbf kernel: C = {C} gamma = {gamma} f1 = {f1_opt}")
        logging.info(classifier.get_params())
        return classifier
    elif kernel == "linear" :
        for i in List_C:
            #Linear kernel doesn't need gamma
            progress(count, len(List_C),'C')
            count+=1
            cls = SVC(kernel='linear', C=i)
            classifier = Pipeline([('vec', vec), ('cls',cls)])
            classifier.fit(X_train, Y_train)
            pred = classifier.predict(X_test)
            f1 = report(Y_test, pred, digits=3, output_dict = True, zero_division = 0).get('macro avg').get('f1-score')
            #If F1 of model is higher than the maximum F1 of previous models, set new maximum F1
            if f1 > f1_opt:
                best_model = classifier
                f1_opt = f1
                C = i
        logging.info(f"linear kernel: C = {C}  f1 = {f1_opt}")
        logging.info(classifier.get_params())
        return classifier



def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def main():
    args = create_arg_parser()
    n1 = args.n1
    n2 = args.n2
    
    if args.tfidf:
        vec = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(n1,n2))
        experiment_name = "SVM+Tf-idf"+str(n1)+"-"+str(n2)+"-"+args.kernel
    else:
        vec = CountVectorizer(tokenizer=word_tokenize,ngram_range=(n1,n2))
        experiment_name = "SVM+CV"+str(n1)+"-"+str(n2)+"-"+args.kernel

    #Create Log file
    try:
        os.mkdir("Logs")
        logging.basicConfig(filename=LOG_DIR+experiment_name+'.log',level=logging.INFO)
    except OSError as error:
        logging.basicConfig(filename=LOG_DIR+experiment_name+'.log', level=logging.INFO)
    

    X_train, Y_train  = read_data(DATA_DIR+'train.csv')
    X_test, Y_test  = read_data(DATA_DIR+'dev.csv')
    classifier = get_optimal_hyperParmeters( args.kernel, X_train,Y_train,  X_test, Y_test, vec)
    saveModel(classifier, experiment_name)



if __name__ == "__main__":
    main()
    

