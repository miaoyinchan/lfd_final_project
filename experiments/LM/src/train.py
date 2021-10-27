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
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.metrics as metrics



DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

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

    parser.add_argument("-t", "--trial", action="store_true", help="Use smaller dataset for parameter optimization")


    args = parser.parse_args()
    return args


def weighted_loss_function(labels, logits):
    pos_weight = tf.constant(0.5)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight))

def load_data(dir, trial=False):

    if trial:
        df_train = pd.read_csv(dir+'/train_opt.csv')
    else:
        df_train = pd.read_csv(dir+'/train.csv')
    

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']

    Y_train = [1 if y=="MISC" else 0 for y in Y_train]
    Y_dev = [1 if y=="MISC" else 0 for y in Y_dev]

    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev

def classifier(X_train, X_dev, Y_train, Y_dev, model_name):


    if model_name =='LONG':
        lm = "allenai/longformer-base-4096"
        max_length = 1024
    else:
        lm = 'bert-base-uncased'
        max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    # loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=5e-5)
    model.compile(loss=weighted_loss_function, optimizer=optim, metrics=METRICS)

    #callbacks
    es = EarlyStopping(monitor="val_prc", patience=2, restore_best_weights=True, mode='max')
    history_logger=tf.keras.callbacks.CSVLogger(LOG_DIR+model_name+"-history.csv", separator=",", append=True)

    model.fit(tokens_train, Y_train, verbose=1, epochs=3 ,batch_size=8, validation_data=(tokens_dev, Y_dev), callbacks=[es, history_logger])
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
    model_name = args.model
    trial = args.trial
    if trial:
        model_name = model_name+"-trial"


    set_log(model_name)

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR, trial)

    #run model
    classifier(X_train,X_dev,Y_train, Y_dev, model_name)

  

if __name__ == "__main__":
    main()