import os
import argparse
import pandas as pd
import joblib


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"

def create_arg_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")

    parser.add_argument("-n1", "--n1", default=1, type=int,
                        help="Ngram Start point")
    
    parser.add_argument("-n2", "--n2", default=1, type=int,
                        help="Ngram End point")


    args = parser.parse_args()
    return args

def load_data(dir):

    """ Return article and label from test data """

    df = pd.read_csv(dir+'/test.csv')
    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    
    args = create_arg_parser()
    n1 = args.n1
    n2 = args.n2


    if args.tfidf:
        experiment_name = "NB+Tf-idf+"+str(n1)+"-"+str(n2)
    else:
        experiment_name = "NB+CV+"+str(n1)+"-"+str(n2)

    

    #Load a Naive Bayes classifier model
    classifier = joblib.load(MODEL_DIR+experiment_name)

    #load data from train-test-dev folder
    X_test, Y_test = load_data(DATA_DIR)

    #Test the model with test set
    Y_pred = classifier.predict(X_test)

    #save results in dataframe
    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred
    

    #save output
    try:
        os.mkdir(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR+experiment_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(OUTPUT_DIR+experiment_name+".csv", index=False)
        
    

if __name__ == "__main__":
    main()