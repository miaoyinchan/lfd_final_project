import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
import pickle
import gc
import os
import csv
import random as python_random
import json
import fasttext
import fasttext.util
import argparse
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer,BertForMaskedLM,BertTokenizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM, Dropout
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline
from itertools import islice
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
from tqdm.keras import TqdmCallback
from tensorflow.keras import backend as K
import keras.metrics as metrics
import joblib


# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
INPUT = '../input/jigsaw-bert-preprocessed-input/'
TEXT_COL = 'comment_text'
MAXLEN = 250

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'





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
    """
    Description:
    
    This method is an arg parser
    
    Return
    
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding",type=str, default='fast',
                        help="Word embedding for LSTM")
    parser.add_argument("-m", "--model",type=str, default='default',
                        help="Name of model")
    parser.add_argument("-b", "--batchsize",type=int, default=16,
                        help="Batchsize")
    parser.add_argument("-l", "--Learningrate",type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("-d", "--dropout",type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("-lay", "--LSTM_layers",type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("-bi", "--bidirectional",action="store_true",
                        help="Add bidirectional LSTM")
    args = parser.parse_args()
    return args



def load_data(dir):

    df_train = pd.read_csv(dir+'/train.csv')

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']

    Y_train = [1 if y=="MISC" else 0 for y in Y_train]
    Y_dev = [1 if y=="MISC" else 0 for y in Y_dev]
    #X_train, Y_train = upscale_minority(X_train,Y_train)
    #writetocsv(X_train,Y_train)
    #X_train, Y_train = readAug(X_train,Y_train)
    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev

def get_embeddings(docs):
    model = FastText(vector_size=50)
    model.build_vocab(docs)
    model.train(docs, epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words)
    return model.wv;

def get_embeddings_glove():
    embeddings_dict = {}
    with open("data/glove.6B.200d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
        return embeddings_dict


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb[word]
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix

def get_emb_matrix2(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    print(embedding_dim)
    print(num_tokens)
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix,args):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    learning_rate = args.Learningrate
    #loss_function = 'binary_crossentropy'
    optim = Adam(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(Y_train[0])
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=True))
    model.add(Dropout(args.dropout, input_shape=(emb_matrix.shape[1],)))
    # Here you should add LSTM layers (and potentially dropout)
    if args.bidirectional:
        lstm = Bidirectional(LSTM(128))
        lstmseq = Bidirectional(LSTM(128, return_sequences=True))
    else:
        lstm = LSTM(128)
        lstmseq = LSTM(128, return_sequences=True)
    

    if args.LSTM_layers > 1:
        for i in range(args.LSTM_layers):
            if i+1 == args.LSTM_layers:
                model.add(lstm)
            else:
                model.add(lstmseq)
    else:
        model.add(lstm)
    #model.add(Bidirectional(LSTM(128,return_sequences=True)))
    #model.add(Bidirectional(LSTM(128)))
    #model.add(Bidirectional(LSTM(128)))
    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="sigmoid"))
    
    # Compile model using our settings, check for accuracy
    model.compile(loss= "binary_crossentropy", optimizer=optim,  metrics=METRICS)
    logging.info(model.summary())
    
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, name, args):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = args.batchsize
    epochs = 50
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Finally fit the model to our data
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, mode='max')
    history_logger = CSVLogger(LOG_DIR+name+"-history.csv", separator=",", append=True)
    #class_weight = {0:0.75,1:0.25}

    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[es, history_logger, TqdmCallback(verbose=2)], batch_size=batch_size, validation_data=(X_dev, Y_dev),class_weight = {0:0.75,1:0.25})
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev")
    saveModel(model,name)
    return model

def saveModel(classifier,experiment_name ):
    #save model
    try:
        os.mkdir(MODEL_DIR)
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)

    except OSError as error:
        joblib.dump(classifier, MODEL_DIR+experiment_name, compress=9)


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
    
def weighted_loss_function(labels, logits):
    pos_weight = tf.constant(0.33)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight))

def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    score = roc_auc_score(Y_test, Y_pred)
    print('ROC AUC: %.3f' % score)
    print(classification_report(Y_test, Y_pred))


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
    logging.info(args)

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR)


    #run model

   # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=200)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    
    if args.embedding == "glove":
        emb = get_embeddings_glove()
        emb_matrix = get_emb_matrix2(voc, emb)
        print("yes")
    else:
       emb = get_embeddings(voc)
       emb_matrix = get_emb_matrix(voc, emb)
       print("yes")
    #print(emb_matrix)
    


    
    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix, args)
    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()
    
    # Train the model
    model = train_model(model, X_train_vect, Y_train, X_dev_vect, Y_dev,model_name, args)

if __name__ == "__main__":
    main()
