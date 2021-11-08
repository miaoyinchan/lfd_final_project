import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import json
import os
import argparse
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import random as python_random
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)



DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"


def create_arg_parser():
    """
    Description:
    
    This method is an arg parser
    
    Return
    
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default= "base", type=str, help="model to test")
    parser.add_argument("-t", "--training_set", default= "aug", type=str, help="training set that was used for training")

    args = parser.parse_args()
    return args


def load_data(data_set,dir):
    """
    Description:
    
    This method load the training and dev set
    
    Return
    
    train, dev and test set with their labels as lists
    """
    if data_set == "aug":

        df_train = pd.read_csv(dir+'/'+'train_aug.csv')

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']
    
    df_test = pd.read_csv(dir+'/test.csv')

    X_test = df_test['article'].ravel().tolist()
    Y_test = df_test['topic']

    Y_train = [1 if y=="MISC" else 0 for y in Y_train]
    Y_dev = [1 if y=="MISC" else 0 for y in Y_dev]
    Y_test = [1 if y=="MISC" else 0 for y in Y_test]

    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    Y_test = tf.one_hot(Y_test,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def save_output(Y_test, Y_pred, model_name):
    """
    Description:
    
    This method saved the predicted labels and the test labels
   """
    Y_test = ["MISC" if y==1 else "CLIMATE" for y in Y_test]
    Y_pred = ["MISC" if y==1 else "CLIMATE" for y in Y_pred]

    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred

    #save output
    try:
        os.mkdir(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
   

def test(X_test, Y_test, model_name):
    """
    Description:
    
    This method predicts the labels of the test set
   
    Return list with predicted label and gold standartd
    """
    #load model
    model_json = f"{model_name}.json"
    model_weight = f"{model_name}.h5"
    json_file = open(MODEL_DIR+model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_DIR+model_weight)
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    Y_pred = loaded_model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), "test"))
    print(classification_report(Y_test, Y_pred))
    return Y_test, Y_pred

    
    

    
    return Y_test, Y_pred

def set_log(model_name):

    #Create Log file
    try:
        os.mkdir(LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
    
        log.setLevel(logging.INFO)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

def main():

    args = create_arg_parser()
    
    model_name = args.model

    set_log(model_name)
    max_length  =  1000

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev,X_test, Y_test = load_data(args.training_set,DATA_DIR)
    vectorizer = TextVectorization(standardize=None, output_sequence_length=1000)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()

    Y_test, Y_pred = test(X_test_vect, Y_test, model_name)
    
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()
