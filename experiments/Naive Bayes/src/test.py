import os
import pandas as pd
import joblib
import utils
import argparse

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ts",
        "--testset",
        default="24",
        type=str,
        help="define the test set. By default it uses the 24th meeting as test set")

    args = parser.parse_args()
    return args

def load_data(dir, testset):

    """ Return article and label from test data """

    if testset == "24":
        df = pd.read_csv(dir+'/test.csv')
    elif testset=="25":
        df = pd.read_csv(dir+'/test_25th.csv')

    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    
    #get parameters for experiments
    _, model_name = utils.get_config()

    #Load a Naive Bayes classifier model
    classifier = joblib.load(utils.MODEL_DIR+model_name)

    #load data from train-test-dev folder
    args = create_arg_parser()
    X_test, Y_test = load_data(utils.DATA_DIR, args.testset)

    #Test the model with test set
    Y_pred = classifier.predict(X_test)

    #save results in dataframe
    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred
    

    #save output
    try:
        os.mkdir(utils.OUTPUT_DIR)
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
        
    

if __name__ == "__main__":
    main()