import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info

import os
import argparse
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.ops.gen_math_ops import mod
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

def create_arg_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-lm",
        "--model",
        default="BERT",
        const="BERT",
        nargs="?",
        choices=[
            "BERT",
            "LONG"

        ],
        help="Select feature from the list",
    )



def weighted_loss_function(target, output):
    pos_weight = tf.constant(0.5)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight))

def load_data(dir):

    df_test = pd.read_csv(dir+'/test.csv')

    X_test = df_test['article'].ravel().tolist()
    Y_test = df_test['topic']

    encoder = LabelBinarizer()
    Y_test = encoder.fit_transform(Y_test)

    
    return X_test, Y_test

def save_output(Y_test, Y_pred, model_name):

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

    if model_name =="BERT":
        lm = 'bert-base-uncased'
        max_length = 512
    elif model_name =='LONG':
        lm = "longformer-base-4096"
        max_length = 1024
    
    lm = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR+model_name)
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    return Y_test, Y_pred

def set_log():

    #Create Log file
    try:
        os.mkdir(LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
    
        log.setLevel(logging.INFO)

    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler('test-logs.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

def main():

    args = create_arg_parser()
    model_name = args.model

    set_log()




    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR)
    Y_test, Y_pred = test(X_train,X_dev,Y_train, Y_dev, lm)
    
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()