import os
import json
import pandas as pd
import joblib


DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"

def get_config():

    """Return model name and paramters after reading it from json file"""
    try:
        location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v).upper() for v in configs.values()]
            model_name = "_".join(vals)
        return model_name
    except FileNotFoundError as error:
        print(error)

def load_data(dir):

    """ Return article and label from test data """

    df = pd.read_csv(dir+'/test.csv')
    X = df['article'].ravel()
    Y = df['topic']
    
    return X,Y

def main():

    
    #get parameters for experiments
    experiment_name = get_config()

    

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