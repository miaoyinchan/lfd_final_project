from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib import pyplot as plt
from wordcloud import WordCloud

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


    args = parser.parse_args()
    return args

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
        df = pd.read_csv(OUTPUT_DIR+"results.csv")
        df = df.append(result, ignore_index=True)
        df.to_csv(OUTPUT_DIR+"results.csv",index=False)
    except FileNotFoundError:
        df = pd.DataFrame(result,index=[0])
        df.to_csv(OUTPUT_DIR+"results.csv",index=False)

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

def cloud(freqs,title):
    '''word cloud genertaed from frquency'''
    plt.figure(figsize=(30,20))
    w = WordCloud(width=3000,height=2400,mode='RGBA',background_color='white',max_words=15000).fit_words(freqs)
    plt.imshow(w)
    plt.axis("off")
    plt.savefig(OUTPUT_DIR+title+'-word-cloud.jpg')

def most_informative_features(classifier,experiment_name, n=100):
    
    """ get most informative features"""
    v = classifier[1].coef_.ravel()
    features = classifier[0].get_feature_names()
    coefs_with_fns = sorted(zip(v, features))
    
    top_misc =  {fn_1:coef_1 for coef_1, fn_1 in coefs_with_fns[:-(n+1):-1]}
    top_climate = {fn_1:abs(coef_1) for coef_1, fn_1 in coefs_with_fns[:n:]}
    
    top_climate_list = list(top_climate.keys())
    top_misc_list = list(top_misc.keys())
    cloud(top_climate,experiment_name+"-"+"climate")
    cloud(top_misc,experiment_name+"-"+"misc")
    
    
    return top_climate_list, top_misc_list


def main():

    
    args = create_arg_parser()
    n1 = args.n1
    n2 = args.n2


    if args.tfidf:
        experiment_name =  "SVM+Tf-idf1-1-linear"
    else:
        experiment_name = "SVM+CV1-1-linear" 

    
    output = pd.read_csv(OUTPUT_DIR+experiment_name+'.csv')
    Y_test = output['Test']
    Y_predict = output['Predict']

    save_results(Y_test, Y_predict, experiment_name)

    classifier = joblib.load(MODEL_DIR+experiment_name)

    #Find top features
    climate_posterior, misc_posterior = most_informative_features(classifier,experiment_name, 100)
    #Save top features in csv file
    df = pd.DataFrame()
    df['Climate'] = climate_posterior
    df['MISC'] = misc_posterior

    df.to_csv(OUTPUT_DIR+experiment_name+"_top_features.csv", index= False)

        



if __name__ == "__main__":
    main()
