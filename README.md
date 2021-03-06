# Binary Text Classification On Imbalanced Data

 One of the most significant developments in the field of Natural Language Processing (NLP) in the recent decade has been the emergence of powerful pre-trained language models. Language models are pre-trained to understand a language so that they can be asked to perform any given task, such as text classification. A real scenario of the NLP world is that imbalanced data often make models biased. It is interesting to know how far these pre-trained language models can deal with an imbalanced dataset. In this work, we have compared pre-trained language model BERT with LongTransformer, Long Short-Term Memory architecture (LSTM), and classical machine learning models in a binary text classifications task on an imbalanced data set. Moreover, data augmentation and downsampling techniques were applied to see if they can help to improve the performance of the models. In an imbalanced dataset with binary class, we find the fine-tuned Longformer model obtained an F1-score of 94.50 for the minority class and a macro F1-score of 96.08.

## Environment Setup

Requires Python 3.8+ and it is recommended to use virtual environment such as virtualenv

Clone this reppository `https://github.com/miaoyinchan/lfd_final_project.git`

To install recquired packagaes:

```
pip install -r requirements.txt

```

In case you cannot download nltk to Peregrine from this command line in data-processing.py

```nltk.download("punkt")```

Please try the following codes in Peregrine  To install NLTK data on Peregrine:

```
python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
```

## Data Pre-Processing

**IMPORTNAT**

* It is assumed that data is present at ***data*** folder
* The following command will **NOT** add augmented data and it is recommended to create a directory named ***train-test-dev*** in the project folder and download the file ***train_aug.csv*** from [here](https://drive.google.com/file/d/1tHI_j5RUNZH8Cx2NIaR_L9oC765s3CSX/view?usp=sharing).

* For data pre-processing, run: `pre-processing.sh`


* To add augmented data from scratch you can run `pre-processing.sh upsampling`  but it can take considerable amount of time (e.g 4-6 hours) to finish.

* Please ensure you have *test.csv* and *test_25th.csv* files in the *train-test-dev* folder

To get a summery of data and create graphical plots, run:

`python3 data-analysis.py`

## Experiments

Download all saved models from [here](https://drive.google.com/drive/folders/1g7D1uaNfiLXztpZqMaM8WWEU00cxeSKL?usp=sharing). 

* Please download all files in *lfd_final_project* folder and execute `model-distribute.sh`

* The bash script will combine all zip files and distribute into respective folders



### Pre-trained Language Model


* Models are available for following parameters:

| Parameter | Value|
|------------- |------------- |
| Pre-trained Model | **BERT**, **LONG** (Longformer) |
| Max Sequence Length | **512**, **1024** (Longformer) |
| Learning rate | **1e-4, 3e-4, 5e-5** |
| Optimizer | **Adam, SGD** |
| Loss function| **Binary** (Binary Crossentropy), **Custom** (Weighted Loss Function) |
| Training-set| **Full** (train.csv), **Resample** (train_aug.csv), **Resample-balance** (train_down.csv)|
| Batch Size| **8** |
| Number of Epochs| **10** |
| Early Stopping Patience| **3** |


* Parameters can be changed at ***experiments/LM/src/config.json*** file
* By default parameters from the best model is given the config file
* An example of a configuration file is given below
    
    ```json

   
    "model": "LONG",
    "max_length" : 1024,
    "learning_rate": 3e-4,
    "epochs": 10,
    "patience": 3,
    "batch_size": 8,
    "loss": "binary",
    "optimizer": "sgd",
    "training-set": "full"


    ```
* Values in between brackets are only given here for explananation. Please use the values presented as bold text into the config.json file to run experiments. 

* To run experiments with Naive Bayes Algorithm, run bash file from ***experiments/Naive Bayes/***

    execute `lm.sh [testset] [--option]`

    * **testset:** use `24` to test the model on data from 24th meeting and
                       `25` will test the model on 25th COP meeting **[Mandatory]**

    * **--option:** use `t` to train, predict, and evaluate a model from scratch. by default it only predict outputs from a saved model and evaluate the result **[OPTIONAL]**
    
    **Example** `lm.sh 25 t` or `lm.sh 24` 

### Naive Bayes


* Models are available for following parameters:

| Parameter | Value|
|------------- |------------- |
| word n-gram range | **1-1** |
| vectorizer | **tf-idf**, **cv** (CountVectorizer) |
| training-set| **Full** (train.csv)|


* Parameters can be changed at ***experiments/Naive Bayes/src/config.json*** file

* To run experiments with Naive Bayes Algorithm, run bash file from ***experiments/Naive Bayes/***

    execute `nb.sh [testset] [--option]`

    * **testset:** use `24` to test the model on data from 24th meeting and
                       `25` will test the model on 25th COP meeting **[Mandatory]**

    * **--option:** use `t` to train, predict, and evaluate a model from scratch. by default it only predict outputs from a saved model and evaluate the result **[OPTIONAL]**
    
    **Example** `nb.sh 25 t` or `nb.sh 24`

### Random Forest

* To run the baseline model using Random Forest algorithm (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data), run bash file from ***experiments/RF/***

    * excecute `run_24.sh` to train the RF model (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data), then test the model with our **test set** (the **24th** COP meeting), and then evaluate the model and print F1 scores of the model to file (experiments/RF/Output/results.csv).

    * excecute `run_25.sh` to train the RF model (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data), then test the model with the **unseen test set** (the **25th** COP meeting), and then evaluate the model and print F1 scores of the model to file (experiments/RF/Output/results.csv).


* If just to test and evaluate the baseline RF model (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data) which has been trained and uploaded in GoogleDrive:

    * The baseline RF model named "RF+Tf-idf_ccp_alpha_0.0".
    * execute `test.py -t -b -ts 24` to test the model with the **test set**, or `test.py -t -b -ts 25` to test the model with the **unseen test set**.
    * execute `evaluate.py -t -b` to get scores of the model.


* To train, test, and evaluate models with different features and parameter values(Count Vector or TF-IDF, with different alphas):

    * execute `train.py -h`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.
### SVM
* To run the baseline model using SVM algorithm, run bash file from ***experiments/SVM/***
    * excecute `svm.sh` to train the SVM model using Countvectors, test the model against our test set and evaluate it.
    * excecute `svm_tfidf.sh` to train the SVM model using TF-IDF, test the model against our test set and evaluate it.
    * excecute `svm_25.sh` to train the SVM model using Countvectors, test the model against the new test set (cop 25)  and evaluate it.
    * excecute `svm_tfidf_25.sh` to train the SVM model using TF-IDF, test the model against the new test set (cop 25) and evaluate it.


### Optimized Linear SVM

* To run the best Linear SVM model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set), run bash file from ***experiments/OptimizedSVM/***

    * excecute `run_24.sh` to train the best Linear SVM model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set), then test the model with our **test set** (the **24th** COP meeting), and then evaluate the model and print F1 scores of the model to file (experiments/OptimizedSVM/Output/results.csv).

    * excecute `run_25.sh` to train the best Linear SVM model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set), then test the model with the **unseen test set** (the **25th** COP meeting), and then evaluate the model and print F1 scores of the model to file (experiments/OptimizedSVM/Output/results.csv).


* If just to test and evaluate the best model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set) which has been trained and uploaded in GoogleDrive:

    * The best Linear SVM model named "model_upsampling_aug_1".
    * execute`test.py -u1 -ts 24` to test the model with the **test set**, or execute`test.py -u1 -ts 25` to test the model with the **unseen test set**.
    * execute `evaluate.py -u1` to get scores of the model.


* To train, test, and evaluate linear SVC models with different feature sets on the full training set:

    * execute `train.py -h`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.

    * Explanation of feature sets:

| Feature set  |      Setting                   |
|--------------|--------------------------------|
| 1 (-f1)      | word-ngram range (1,3)         |
| 2 (-f2)      | char-5-gram                    |
| 3 (-f3)      | stop words                     |
| 4 (-f4)      | POS tag                        |
| 5 (-f5)      | Lemmetizer                     |
| 6 (-f6)      | NER tag                        |
| 7 (-f7)      | word-ngram range (1,4)         |
| 8 (-f8)      | word(1,3)+char-5gram           |
| 9 (-f9)      | word(1,3)+stopwords            |
| 10 (-f10)    | word(1,3)+char-5gram+stopwords |
| 11 (-f11)    | word-ngram range (2,5)         |


* To tune the linear SVC models:

    * execute `tuning.py`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.


* To train, tune, test, and evaluate linear SVC models on the augmented training set:

    * execute `aug_train.py -h`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.


* To train, tune, test, and evaluate linear SVC models on the fixed sequence length training set:

    * execute `fixed_sq_train.py`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.


### LSTM   

* In order to run the experiment, Please dowload pre-trained word vector from [here](https://nlp.stanford.edu/data/glove.6B.zip). 

* Create a directory names *data* in  *experiments/LSTM/src/data/* and copy the *glove.6B.200d.txt* file from downloaded folder

* To simply test and evaluate an existing model,  
run `test.py -m <modelname> -t <traininge set> -ts <test set>` and `evaluate.py -e <modelname>`

    * The test the LSTM model named *aug_model_model*.
    * execute`test.py -m aug_model_model -t train_aug -ts test.csv` to get predictions from out test set.
    * or `test.py -m aug_model_model -t train_aug -ts test_25th.csv` to use the 25th cop meeting a test set
    * then, execute `evaluate.py -e aug_model_model` to get scores of the model.

* **IMPORTANT**: We find that sometimes keras does not load the weights from saved model and initialize random weights which produce very poor result. We have followed everything from kears documentations but still it does not work. Besides, we have tried solutions form [here](https://github.com/keras-team/keras/issues/4875). It works fine in peregrine but not in our pc.
In case, if it does not work, please run `train_test_evaluate.py` from ***experiments/LSTM/src*** which trains the model, predict outputs and evaluate it.
* To train the model using different parameters, use the --h command of train.py

* run `lstm.sh` from ***experiments/LSTM/src*** which trains the model, predict outputs and evaluate it. You can also change the arguments in the bash script to try different models. 




 * Please use following model names and corresponding training sets to run experiments:

    | Models            | Training data  |
    |-------------------|----------------|
    | aug_model_model | train_aug      |
    | base_model      | train          |
    | down_model      | train_down     |





