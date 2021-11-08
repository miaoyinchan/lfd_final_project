from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import utils


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
        df = pd.read_csv(utils.OUTPUT_DIR+"results.csv")
        df = df.append(result, ignore_index=True)
        df.to_csv(utils.OUTPUT_DIR+"results.csv",index=False)
    except FileNotFoundError:
        df = pd.DataFrame(result,index=[0])
        df.to_csv(utils.OUTPUT_DIR+"results.csv",index=False)

    # save the confusion matrix of the model in png file
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(utils.OUTPUT_DIR+"{}.png".format(experiment_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))

    print("\nClassification Report\n")
    print(classification_report(Y_test,Y_pred))


def find_top_features(classifier, n):

    """Return n most top features per class"""

    prob = classifier[1].feature_log_prob_
    features = classifier[0].get_feature_names_out()
    climate_posterior = {f:p for f,p in zip(features,prob[0])}
    misc_posterior = {f:p for f,p in zip(features,prob[1])}

    #Sort features using their posterior probablitiy
    climate_posterior_sorted = {k:v for k,v in sorted(climate_posterior.items(), key=lambda item: item[1], reverse= True)}
    misc_posterior_sroted = {k:v for k,v in sorted(misc_posterior.items(), key=lambda item: item[1], reverse= True)}

    #get top n features
    climate_posterior_top_features = list(climate_posterior_sorted.keys())[:n]
    misc_posterior_top_features = list(misc_posterior_sroted.keys())[:n]

    return climate_posterior_top_features, misc_posterior_top_features


def main():

    
    #get parameters for experiments
    _, model_name = utils.get_config()
    
    output = pd.read_csv(utils.OUTPUT_DIR+model_name+'.csv')
    Y_test = output['Test']
    Y_predict = output['Predict']

    save_results(Y_test, Y_predict, model_name)

    #Load a Naive Bayes classifier model
    classifier = joblib.load(utils.MODEL_DIR+model_name)

    #Find top features
    climate_posterior, misc_posterior = find_top_features(classifier, 100)

    #Save top features in csv file
    df = pd.DataFrame()
    df['Climate'] = climate_posterior
    df['MISC'] = misc_posterior

    df.to_csv(utils.OUTPUT_DIR+model_name+"_top_features.csv", index= False)

        



if __name__ == "__main__":
    main()