# Binary Text Classification On Imbalanced Data

**Add abstract here**

## Environment Setup

Requires Python 3.6+ and it is recommended to use virtual environment such as virtualenv

Clone this reppository `https://github.com/miaoyinchan/lfd_final_project.git`

To install recquired packagaes:

`pip install -r requirements.txt`


## Data Pre-Processing

For data pre-processing, run:

`./pre-processing.sh`

* It is assumed that data is present at ***data*** folder
* This command will **NOT** add augmented data and it is recommended to create a directory named ***train-test-dev*** in the project folder and download the file ***train_aug.csv*** from here(add link)
* Otherwise, run `./pre-processing.sh upsampling` to add augmented data from scratch but it can take considerable amount of time (e.g 4-6 hours) to finish

To get a summery of data and create graphical plots, run:

`python3 data-analysis.py`

## Experiments

Download all saved models from here(add link). To distribute models in respective folders, **unzip** the downloaded file and run:

`python3 model-distribute.py`


### Pre-trained Language Model

* To run experiments with Pre-trained Language Models, run bash file from ***experiment/LM/***

    * excecute `run.sh train` to train, test, and evaluate a model from scratch
    * excecute `run.sh test` to test and evaluate a saved models
    * excecute `run.sh eval` to evaluate a saved models

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


* Parameters can be changed at ***experiment/LM/src/config.json*** file 
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

* To run experiments with Naive Bayes Algorithm, run bash file from ***experiment/Naive Bayes/***

    * excecute `run.sh train` to train, test, and evaluate a model from scratch
    * excecute `run.sh test` to test and evaluate a saved models
    * excecute `run.sh eval` to evaluate a saved models

* Models are available for following parameters:

| Parameter | Value|
|------------- |------------- |
| word n-gram range | **1-1** |
| vectorizer | **tf-idf**, **cv** (CountVectorizer) |
| training-set| **Full** (train.csv)|


* Parameters can be changed at ***experiment/Naive Bayes/src/config.json*** file