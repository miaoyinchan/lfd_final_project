import logging

# get TF logger for pre-trained transformer model
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info

import random as python_random
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.losses import BinaryCrossentropy
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback

import utils



def f1_score(y_true, y_pred): 

    """Return F1 score of CLIMATE class"""
    
    def recall_m(y_true, y_pred):

        """Return recall score of CLIMATE class"""

        #count the number of correct CLIMATE prediction
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        #count number of true CLIMATE entries
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):

        """Return precision score of CLIMATE class"""

        #count the number of correct CLIMATE prediction
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        #count number of entries predicted as CLIMATE
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    #get precision and recall score of postive class
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def weighted_loss_function(labels, logits):

    pos_weight = tf.constant(0.33)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight))

def load_data(dir, training_set):

    """Return appropriate training and validation sets reading from csv files"""

    if training_set.lower()=="trial":
        df_train = pd.read_csv(dir+'/train_opt.csv')
    elif training_set.lower()=="resample":
        df_train = pd.read_csv(dir+'/train_aug.csv')
    elif training_set.lower()=="resample-balance":
        df_train = pd.read_csv(dir+'/train_down.csv')
        df_train = df_train[:-1] #remove one sample to escape longformer's incompatibility with shapes
    else:
        df_train = pd.read_csv(dir+'/train.csv')
    

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']

    #change MISC as 0 and CLIMATE as 1 to allow tensorflow one hot encoding
    Y_train = [0 if y=="MISC" else 1 for y in Y_train]
    Y_dev = [0 if y=="MISC" else 1 for y in Y_dev]

    #convert Y into one hot encoding
    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev

def classifier(X_train, X_dev, Y_train, Y_dev, config, model_name):

    """Train and Save model for test and evaluation"""

    #set random seed to make results reproducible  
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
    learning_rate =  config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]

    if config["loss"].upper() == "CUSTOM":
        loss_function = weighted_loss_function
    elif config["loss"].upper() == "BINARY":
        loss_function = BinaryCrossentropy(from_logits=True)

    if config['optimizer'].upper() == "ADAM":
        optim = Adam(learning_rate=learning_rate)
    elif config['optimizer'].upper() == "SGD":
        optim = SGD(learning_rate=learning_rate)

    if config["model"].upper() =='LONG':
        lm = "allenai/longformer-base-4096"
    elif config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
        
    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    
    #transform raw texts into model input 
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
   
    #change the data type of model inputs to int32 
    if config["model"] =='LONG':
        tokens_train = utils.change_dtype(tokens_train)
        tokens_dev = utils.change_dtype(tokens_dev)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy',f1_score])

    #callbacks for ealry stopping and saving model history
    es = EarlyStopping(monitor="val_f1_score", patience=patience, restore_best_weights=True, mode='max')
    history_logger = CSVLogger(utils.LOG_DIR+model_name+"-HISTORY.csv", separator=",", append=True)

    #train models
    model.fit(tokens_train, Y_train, verbose=0, epochs=epochs,batch_size= batch_size, validation_data=(tokens_dev, Y_dev), callbacks=[es, history_logger, TqdmCallback(verbose=2)])
    
    #save models in directory
    model.save_pretrained(save_directory=utils.MODEL_DIR+model_name)


def set_log(model_name):

    #Create Log file to save info
    try:
        os.mkdir(utils.LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
        log.setLevel(logging.INFO)

    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs info
    fh = logging.FileHandler(utils.LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    #get parameters for experiments
    config, model_name = utils.get_config()
    
    if config['training-set'] != 'trial':
        model_name = model_name+"_"+str(config['seed'])

    set_log(model_name)

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(utils.DATA_DIR, config["training-set"])

    #run model
    classifier(X_train,X_dev,Y_train, Y_dev, config, model_name)

  

if __name__ == "__main__":
    main()
