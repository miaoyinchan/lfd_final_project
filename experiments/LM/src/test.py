import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import json
import os
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import random as python_random


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

def change_dtype(tokens):

    """Return model inputs after changing data type to int32"""

    tokens['input_ids'] = tokens['input_ids'].astype('int32')
    tokens['input_ids'] = tokens['input_ids'].astype('int32')

    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    
    return tokens

def get_config():

    """Return model name and paramters after reading it from json file"""

    try:
        location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v).upper() for v in configs.values()]
            model_name = "_".join(vals[:-1])
        return configs, model_name
    except FileNotFoundError as error:
        print(error)


def load_data(dir):

    """Return test sets reading from csv files"""

    df_test = pd.read_csv(dir+'/test.csv')
    X_test = df_test['article'].ravel().tolist()
    Y_test = df_test['topic']

    #change MISC as 0 and CLIMATE as 1 to allow tensorflow one hot encoding
    Y_test = [0 if y=="MISC" else 1 for y in Y_test]
    
    #convert Y into one hot encoding
    Y_test = tf.one_hot(Y_test, depth=2)
    
    return X_test, Y_test

def save_output(Y_test, Y_pred, model_name):

    """save models prediction as csv file"""

    #convert 0 as MISC and 1 as CLIMATE
    Y_test = ["MISC" if y==0 else "CLIMATE" for y in Y_test]
    Y_pred = ["MISC" if y==0 else "CLIMATE" for y in Y_pred]

    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred

    #save output in directory
    try:
        os.mkdir(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(OUTPUT_DIR+model_name+".csv", index=False)
   

def test(X_test, Y_test, config, model_name):

    """Return models prediction"""

    #set random seed to make results reproducible 
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
   
    if config["model"].upper() =='LONG':
        lm = "allenai/longformer-base-4096"
    elif config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
        
    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    #transform raw texts into model input 
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    #change the data type of model inputs to int32 
    if config["model"] =='LONG':
        tokens_test = change_dtype(tokens_test)

    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR+model_name)

    #get model's prediction
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    return Y_test, Y_pred

def set_log(model_name):

    #create log file
    try:
        os.mkdir(LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
    
        log.setLevel(logging.INFO)
    
    #create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #create file handler which logs info
    fh = logging.FileHandler(LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #get parameters for experiments
    config, model_name = get_config()
    
    if config['experiment'] != 'trial':
        model_name = model_name+"_"+str(config['seed'])

    #set log settings
    set_log(model_name)

    #load data from train-test-dev folder
    X_train, Y_train = load_data(DATA_DIR)
    Y_test, Y_pred = test(X_train, Y_train, config, model_name)
    
    #save output in directory
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()