import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import numpy as np

import os
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import SGD
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

#Loding pretrained bert layer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)




DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

def load_data(dir):

    df_train = pd.read_csv(dir+'/train_opt.csv')

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

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

MAX_LEN = 64

def main():
    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR)
    # Loading tokenizer from the bert layer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocab_file, do_lower_case)
    train_input = bert_encode(X_train, tokenizer, max_len=MAX_LEN)
    # encode  test set 
    test_input = bert_encode(X_dev, tokenizer, max_len= MAX_LEN )
    train_labels = Y_train
    print(train_labels)
    # first define input for token, mask and segment id  
    from tensorflow.keras.layers import  Input
    input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
    Y_dev_bin = encoder.fit_transform(Y_dev)

    #  output  
    from tensorflow.keras.layers import Dense
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])  
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)   

    # intilize model
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # train
    train_history = model.fit(
    train_input, y_train,
    validation_split=0.2,
    epochs=2,
    batch_size=32)



if __name__ == "__main__":
    main()
