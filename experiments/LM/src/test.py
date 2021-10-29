import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import json
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
from tqdm.keras import TqdmCallback


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

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

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trial", action="store_true", help="Use smaller dataset for parameter optimization")
    args = parser.parse_args()
    return args


def load_data(dir):

    df_test = pd.read_csv(dir+'/test.csv')

    X_test = df_test['article'].ravel().tolist()
    Y_test = df_test['topic']

    Y_test = [0 if y=="MISC" else 1 for y in Y_test]
    Y_test = tf.one_hot(Y_test, depth=2)
    
    return X_test, Y_test

def save_output(Y_test, Y_pred, model_name):

    Y_test = ["MISC" if y==0 else "CLIMATE" for y in Y_test]
    Y_pred = ["MISC" if y==0 else "CLIMATE" for y in Y_pred]

    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred

    #save output
    try:
        os.mkdir(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
   

def test(X_test, Y_test, config, model_name):

    
    max_length  =  config['max_length']
   
    if config["model"] =='LONG':
        lm = "allenai/longformer-base-4096"
    else:
        lm = 'bert-base-uncased'
        
    
    lm = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR+model_name)
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
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
    config, model_name = get_config()
    trial = args.trial
    if trial:
        model_name = model_name+"-trial"


    set_log(model_name)

    #load data from train-test-dev folder
    X_train, Y_train = load_data(DATA_DIR)
    Y_test, Y_pred = test(X_train, Y_train, config, model_name)
    
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()