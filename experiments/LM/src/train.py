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



    args = parser.parse_args()
    return args


def weighted_loss_function(target, output):
    pos_weight = tf.constant(0.5)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight))

def load_data(dir):

    df_train = pd.read_csv(dir+'/train.csv')

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']

    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)
    Y_dev = encoder.fit_transform(Y_dev)
    
    return X_train, Y_train, X_dev, Y_dev

def classifier(X_train, X_dev, Y_train, Y_dev, model_name):


    if model_name =="BERT":
        lm = 'bert-base-uncased'
        max_length = 512
    elif model_name =='LONG':
        lm = "allenai/longformer-base-4096"
        max_length = 1024

    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    # loss_function = weighted_loss_function
    optim = Adam(learning_rate=5e-5)
    model.compile(loss=weighted_loss_function, optimizer=optim, metrics=['accuracy'])
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.set
    model.fit(tokens_train, Y_train, verbose=1, epochs=3 ,batch_size=8, validation_data=(tokens_dev, Y_dev), callbacks=[es])
    
    model.save_pretrained(save_directory=MODEL_DIR+model_name)



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

    #run model
    classifier(X_train,X_dev,Y_train, Y_dev, model_name)

  

if __name__ == "__main__":
    main()