import os
import argparse
import pandas as pd
import joblib


DATA_DIR = '../../train-test-dev/'
MODEL_DIR = "Saved_Models/"
OUTPUT_DIR = "Output/"

def create_arg_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")

    parser.add_argument("-n1", "--n1", default=1, type=int,
                        help="Ngram Start point")
    
    parser.add_argument("-n2", "--n2", default=1, type=int,
                        help="Ngram End point")
    parser.add_argument("-ts", "--test_set", default='24', type=str,
                        help="Which test set to use")


    args = parser.parse_args()
    return args

def load_data(dir, testset):

    """ Return article and label from test data """
    if testset=="24":
        df = pd.read_csv(dir+'/test.csv')
        X = df['article'].ravel()
        Y = df['topic']
    
        Y = [1 if y=="MISC" else 0 for y in Y]
    
        return X,Y
    elif testset=="25":
        df = pd.read_csv(dir+'/test_25th.csv')
        X = df['article'].ravel()
        Y = df['topic']
    
        Y = [1 if y=="MISC" else 0 for y in Y]
    
        return X,Y

    
def main():

    
    args = create_arg_parser()
    n1 = args.n1
    n2 = args.n2


    if args.tfidf:
        experiment_name = "SVM+Tf-idf1-1-linear"
    else:
        experiment_name = "SVM+CV1-1-linear"

    
    classifier = joblib.load(MODEL_DIR+experiment_name)
    test_set = args.test_set

    #load data from train-test-dev folder
    X_test, Y_test = load_data(DATA_DIR, test_set)

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
