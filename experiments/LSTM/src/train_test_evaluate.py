import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import numpy as np 
import pandas as pd 
import os

from sklearn.metrics import roc_auc_score

import random as python_random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM, Dropout
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from gensim.models.fasttext import FastText
from tqdm.keras import TqdmCallback






# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


MAXLEN = 1000

DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"



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
    parser.add_argument("-b", "--batchsize",type=int, default=8,
                        help="Batchsize")
    parser.add_argument("-l", "--Learningrate",type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("-d", "--dropout",type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("-rec", "--recurrent",action="store_true",
                        help="Recurrent dropout")
    parser.add_argument("-lay", "--LSTM_layers",type=int, default=1, choices=range(1,3),
                        help="Number of LSTM layers")
    parser.add_argument("-bi", "--bidirectional",action="store_true",
                        help="Add bidirectional LSTM")
    parser.add_argument("-tr", "--training_data",type=str, default='aug',
                        help="Specify training data" )
    parser.add_argument("-i", "--input_file", type=str, default='train.csv',
                        help="Input file for training")
    parser.add_argument("-ts", "--test_set", action="store_true",
                        help="Use cop 25 test set")
    args = parser.parse_args()
    return args



def load_data(dir, tfile, test):
    """
    Description:
    
    This method load the training and dev set
    
    Return
    
    train and dev set with their labels as lists
    """
    df_train = pd.read_csv(dir+'/'+tfile)

    X_train = df_train['article'].ravel().tolist()
    Y_train = df_train['topic']

    df_dev = pd.read_csv(dir+'/dev.csv')

    X_dev = df_dev['article'].ravel().tolist()
    Y_dev = df_dev['topic']
    
    df_test = pd.read_csv(dir+'/'+test)

    X_test = df_test['article'].ravel().tolist()
    Y_test = df_test['topic']

    Y_train = [1 if y=="MISC" else 0 for y in Y_train]
    Y_dev = [1 if y=="MISC" else 0 for y in Y_dev]
    Y_test = [1 if y=="MISC" else 0 for y in Y_test]

    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    Y_test = tf.one_hot(Y_test,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def get_embeddings(docs):
    """
    Description:
    
    This method get Fasttext embeddings
   
    Return word embeddings
    """
    model = FastText(vector_size=50)
    model.build_vocab(docs)
    model.train(docs, epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words)
    return model.wv

def get_embeddings_glove():
    """
    Description:
    
    This method get GLOVE embeddings
   
    Return word embeddings
    """
    embeddings_dict = {}
    with open("data/glove.6B.200d.txt", 'r',  encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
        return embeddings_dict


def get_emb_matrix(voc, emb):
    """
    Description:
    
    This method gets embedding matrix given vocab and the fasttext embeddings
   
    Return embedding matrix 
    """

    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
   
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
    """
    Description:
    
    This method gets embedding matrix given vocab and the glove embeddings
   
    Return embedding matrix 
    """

    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    embedding_dim = len(emb["the"])

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
    """
    Description:
    
    This method creates the Keras model to use
   
    Return  model
    """

    learning_rate = args.Learningrate
    optim = SGD(learning_rate=learning_rate)

    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(Y_train[0])

    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=True))
    if args.recurrent == None:
        model.add(Dropout(args.dropout, input_shape=(emb_matrix.shape[1],)))

    if args.bidirectional:
        if args.recurrent:
            lstm = Bidirectional(LSTM(128,recurrent_dropout=args.dropout))
            lstmseq = Bidirectional(LSTM(128,recurrent_dropout=args.dropout, return_sequences=True))
        else:
            lstm = Bidirectional(LSTM(128))
            lstmseq = Bidirectional(LSTM(128,  return_sequences=True))
    else:
        if args.recurrent:
            lstm = LSTM(128,recurrent_dropout=args.dropout)
            lstmseq = LSTM(128, return_sequences=True,recurrent_dropout=args.dropout)
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

    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="sigmoid"))
    
    # Compile model using our settings, check for accuracy
    model.compile(loss= "binary_crossentropy", optimizer=optim, metrics=['accuracy'])
    logging.info(model.summary())
    
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, name, args):
    """
    Description:
    
    This method trains the model here
   
    Return  model
    """

    verbose = 1
    batch_size = args.batchsize
    epochs = 10

    es = EarlyStopping(monitor="", patience=2, restore_best_weights=True, mode='max')
    history_logger = CSVLogger(LOG_DIR+name+"-history.csv", separator=",", append=True)
    if args.training_data == "aug":
        class_weight = {0:0.60,1:0.40}
    elif args.training_data == "base":
        class_weight = {0:0.85,1:0.15}
    else:
        class_weight = {0:1,1:1}

    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[es, history_logger, TqdmCallback(verbose=2)], batch_size=batch_size, validation_data=(X_dev, Y_dev),class_weight = {0:0.85,1:0.15})

    test_set_predict(model, X_dev, Y_dev, "dev")
    saveModel(model,name)
    return model

def saveModel(classifier,experiment_name ):
    """
    Description:
    
    This method saves our model
    """

    try:
        os.mkdir(MODEL_DIR)
        model_json = classifier.to_json()
        with open(MODEL_DIR+'/'+experiment_name+"_model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        classifier.save_weights(MODEL_DIR+'/'+experiment_name+"_model.h5")
        print("Saved model to disk")
 

    except OSError as error:
        model_json = classifier.to_json()

        with open(MODEL_DIR+'/'+experiment_name+"_model.json", "w") as json_file:
            json_file.write(model_json)
        classifier.save_weights(MODEL_DIR+'/'+experiment_name+"_model.h5")
        print("Saved model to disk")


  

def test_set_predict(model, X_test, Y_test, ident):
    """
    Description:
    
    This method predicts the labels of the dev set
   
    Return list with predicted label and gold standartd
    """

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
    return Y_test, Y_pred


def set_log(model_name):
    """
    Description:
    
    This method creates a Log file
   
    """
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

def save_output(Y_test, Y_pred, model_name):
    """
    Description:
    
    This method saved the predicted labels and the test labels
   """
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
   
def save_results(Y_test, Y_pred, experiment_name):
   
    
    ''' save results (accuracy, precision, recall, and f1-score) in csv file and plot confusion matrix '''

   
    test_report =  classification_report(Y_test,Y_pred,output_dict=True,digits=4)

    result = {"experiment":experiment_name}

    labels = list(test_report.keys())[:2]

    for label in labels:
        result["precision-"+label] = test_report[label]['precision']
        result["recall-"+label] = test_report[label]['recall']
        result["f1-"+label] = test_report[label]['f1-score']
        
    
    result['accuracy'] = test_report['accuracy'] 
    result['macro f1-score'] = test_report['macro avg']['f1-score']

    try:
        df = pd.read_csv(OUTPUT_DIR+experiment_name+"_results.csv")
        df = df.append(result, ignore_index=True)
        df.to_csv(OUTPUT_DIR+experiment_name+"_results.csv",index=False)
    except FileNotFoundError:
        df = pd.DataFrame(result,index=[0])
        df.to_csv(OUTPUT_DIR+experiment_name+"_results.csv",index=False)

    # save the confusion matrix of the model in png file
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(OUTPUT_DIR+"{}.png".format(experiment_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))

    print("\nClassification Report\n")
    print(classification_report(Y_test,Y_pred))

def main():

    args = create_arg_parser()
    model_name = args.model
    set_log(model_name)
    print(args)
    if args.test_set:
        test_set = "test_25th.csv"
    else:
        test_set = "test.csv"

    #load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev,X_test, Y_test = load_data(DATA_DIR, args.input_file, test_set)

   # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=MAXLEN)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    
    if args.embedding == "glove":
        emb = get_embeddings_glove()
        emb_matrix = get_emb_matrix2(voc, emb)
    else:
       emb = get_embeddings(voc)
       emb_matrix = get_emb_matrix(voc, emb)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix, args)
    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
    
    # Train the model
    model = train_model(model, X_train_vect, Y_train, X_dev_vect, Y_dev,model_name, args)
    test, pred = test_set_predict(model, X_test_vect, Y_test, "test")
    
    # Test and evaluate
    save_output(test, pred, model_name)
    save_results(test, pred, model_name)

if __name__ == "__main__":
    main()
