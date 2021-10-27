from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib

DATA_DIR = '../../../train-test-dev/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"

def create_arg_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-lm",
        "--model",
        default="BERT",
        const="BERT",
        nargs="?",
        choices=[
            "BERT",
            "LONG"

        ],
        help="Select feature from the list",
    )



    args = parser.parse_args()
    return args

def save_results(Y_test, Y_pred, model_name):
    
    
    ''' save results (accuracy, precision, recall, and f1-score) in csv file and plot confusion matrix '''

   
    test_report =  classification_report(Y_test,Y_pred,output_dict=True,digits=4)

    result = {"Model": model_name}

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
    plt.savefig(OUTPUT_DIR+"{}.png".format(model_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))

    print("\nClassification Report\n")
    print(classification_report(Y_test,Y_pred))





def main():

    
    args = create_arg_parser()
    model_name = args.model
    
    output = pd.read_csv(OUTPUT_DIR+model_name+'.csv')
    Y_test = output['Test']
    Y_predict = output['Predict']

    save_results(Y_test, Y_predict, model_name)



if __name__ == "__main__":
    main()