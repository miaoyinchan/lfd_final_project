from math import e
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "Output/"


def save_results(Y_test, Y_pred, experiment_name):
    ''' save results (accuracy, precision, recall, and f1-score)
    in csv file and plot confusion matrix '''

    test_report =  classification_report(Y_test,Y_pred,output_dict=True,digits=4)

    result = {"experiment":experiment_name}

    labels = list(test_report.keys())[:3]

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


def main():

    experiment_names = ['ccp_alpha_0.0', 'ccp_alpha_0.01',
                        'ccp_alpha_0.001', 'ccp_alpha_0.0001']
    for experiment_name in experiment_names:
        output = pd.read_csv(OUTPUT_DIR+experiment_name+'.csv')
        Y_test = output['Test']
        Y_predict = output['Predict']

        save_results(Y_test, Y_predict, experiment_name)


if __name__ == "__main__":
    main()
