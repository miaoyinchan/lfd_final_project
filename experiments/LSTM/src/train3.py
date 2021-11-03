import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
import pickle
import gc

BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
INPUT = '../input/jigsaw-bert-preprocessed-input/'
TEXT_COL = 'comment_text'
MAXLEN = 250

DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"

os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(BERT_FP)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat
    


def main():
    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(DATA_DIR)
    embedding_matrix = get_bert_embed_matrix()
