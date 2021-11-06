# Binary Text Classification On Imbalanced Data

**Add abstract here**

## Environment Setup

Requires Python 3.6+ and it is recommended to use virtual environment such as virtualenv

Clone this reppository `https://github.com/miaoyinchan/lfd_final_project.git`

To install recquired packagaes:

```
pip install -r requirements.txt`
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

For data pre-processing, run:

`pre-processing.sh`

* It is assumed that data is present at ***data*** folder
* This command will **NOT** add augmented data and it is recommended to create a directory named ***train-test-dev*** in the project folder and download the file ***train_aug.csv*** from [here](https://drive.google.com/file/d/1tHI_j5RUNZH8Cx2NIaR_L9oC765s3CSX/view?usp=sharing).
* Otherwise, run `pre-processing.sh upsampling` to add augmented data from scratch but it can take considerable amount of time (e.g 4-6 hours) to finish

To get a summery of data and create graphical plots, run:

`python3 data-analysis.py`

## Experiments

Download all saved models from [here](https://drive.google.com/drive/folders/1g7D1uaNfiLXztpZqMaM8WWEU00cxeSKL?usp=sharing). To distribute models in respective folders, **unzip** the downloaded file and run:

`python3 model-distribute.py`


### Pre-trained Language Model

* To run experiments with Pre-trained Language Models, run bash file from ***experiments/LM/***

    * **option:** use `train` to train, test, and evaluate a model from scratch, 
                      `test` to test and evaluate a saved model, 
                      `evaluate` to evaluate a saved model
                    
    * **testset:** use `24` to test the model on data from 24th meeting and
                       `25` will test the model on 25th COP meeting
    
    **Example** `run.sh train 25` 

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
    "learning_rate": 5e-5,
    "epochs": 10,
    "patience": 3,
    "batch_size": 8,
    "loss": "custom",
    "optimizer": "adam",
    "training-set": "resample-balance"


    ```
* Values in between brackets are only given here for explananation. Please use the values presented as bold text into the config.json file to run experiments. 

### Naive Bayes

* To run experiments with Naive Bayes Algorithm, run bash file from ***experiments/Naive Bayes/***

    execute `run.sh [option] [testset]`


    * **option:** use `train` to train, test, and evaluate a model from scratch, 
                      `test` to test and evaluate a saved model, 
                      `evaluate` to evaluate a saved model
                    
    * **testset:** use `24` to test the model on data from 24th meeting and
                       `25` will test the model on 25th COP meeting
    
    **Example** `run.sh train 25` 

* Models are available for following parameters:

| Parameter | Value|
|------------- |------------- |
| word n-gram range | **1-1** |
| vectorizer | **tf-idf**, **cv** (CountVectorizer) |
| training-set| **Full** (train.csv)|


* Parameters can be changed at ***experiments/Naive Bayes/src/config.json*** file

### Random Forest

* To run the baseline model using Random Forest algorithm (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data), run bash file from ***experiments/RF/***

    * excecute `run.sh` to train the RF model (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data), then test the model with test set, and then evaluate the model and print F1 scores of the model to file (experiments/RF/Output/results.csv).


* If just to test and evaluate the baseline RF model (with the best alpha value ccp_alpha=0.0, using TF-IDF features, trained on the full training data) which has been trained and uploaded in GoogleDrive:

    * The baseline RF model named "RF+Tf-idf_ccp_alpha_0.0".
    * execute `test.py -t -b` to test the model.
    * execute `evaluate.py -t -b` to get scores of the model.


* To train, test, and evaluate models with different features and parameter values(Count Vector or TF-IDF, with different alphas):

    * execute `train.py -h`, `test.py -h`, `evaluate.py -h` to see all command line arguments, and choose the desired option.


### Optimized Linear SVM

* To run the best Linear SVM model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set), run bash file from ***experiments/OptimizedSVM/***

    * excecute `run.sh` to train the best Linear SVM model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set), then test the model with test set, and then evaluate the model and print F1 scores of the model to file (experiments/OptimizedSVM/Output/results.csv).


* If just to test and evaluate the best model (with the best C value c=1.0, using TF-IDF vectors, and word ngram range(1,3) features, trained on augmented training set) which has been trained and uploaded in GoogleDrive:

    * The best Linear SVM model named "model_upsampling_aug_1".
    * execute`test.py -u1` to test the model.
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
