import json
import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info

import random as python_random
import numpy as np
import os
import argparse
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.losses import BinaryCrossentropy
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback

DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"


physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def change_dtype(tokens):

    tokens['input_ids'] = tokens['input_ids'].astype('int32')
    tokens['input_ids'] = tokens['input_ids'].astype('int32')

    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    
    return tokens

def get_config():

    try:
        location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v) for v in configs.values()]
            model_name = "_".join(vals)
        return configs, model_name
    except FileNotFoundError as error:
        print(error)


def f1_score(y_true, y_pred): 

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", default= 1234, type=int, help="select seed")

    args = parser.parse_args()
    return args


def weighted_loss_function(labels, logits):
    pos_weight = tf.constant(0.33)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight))

def load_data(dir, experiment):

    if experiment=="trial":
        df_train = pd.read_csv(dir+'/train_opt.csv')
    elif experiment=="resample":
        df_train = pd.read_csv(dir+'/train_aug.csv')
    else:
        df_train = pd.read_csv(dir+'/train.csv')
    

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']

    Y_train = [0 if y=="MISC" else 1 for y in Y_train]
    Y_dev = [0 if y=="MISC" else 1 for y in Y_dev]

    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev

def classifier(X_train, X_dev, Y_train, Y_dev, config, model_name):

    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    max_length  =  config['max_length']
    learning_rate =  config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]

    if config["loss"] == "custom":
        loss_function = weighted_loss_function
    else:
        loss_function = BinaryCrossentropy(from_logits=True)

    if config['optimizer'] == "Adam":
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)

    if config["model"] =='XLNet':
        lm = "xlnet-base-cased"
    elif config["model"] =='LONG':
        lm = "allenai/longformer-base-4096"
    else:
        lm = 'bert-base-uncased'
        

    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
   
    if config["model"] =='LONG':
        tokens_train = change_dtype(tokens_train)
        tokens_dev = change_dtype(tokens_dev)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy',f1_score])

    #callbacks
    es = EarlyStopping(monitor="val_f1_score", patience=patience, restore_best_weights=True, mode='max')
    history_logger = CSVLogger(LOG_DIR+model_name+"-history.csv", separator=",", append=True)

    model.fit(tokens_train, Y_train, verbose=0, epochs=epochs,batch_size= batch_size, validation_data=(tokens_dev, Y_dev), callbacks=[es, history_logger, TqdmCallback(verbose=2)])
    model.save_pretrained(save_directory=MODEL_DIR+model_name)


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
    seed = args.seed

    config, model_name = get_config()
    config['seed'] = seed


    set_log(model_name)

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR, config["experiment"])

    #run model
    classifier(X_train,X_dev,Y_train, Y_dev, config, model_name)

  

if __name__ == "__main__":
    main()
