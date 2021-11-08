#!/usr/bin/env python

"""Plot word cloud of the best linear svm model"""

import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud


MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"


def cloud(freqs, title):
    '''word cloud genertaed from frquency'''
    plt.figure(figsize=(30,20))
    w = WordCloud(width=3000,height=2400,mode='RGBA',background_color='white',max_words=15000).fit_words(freqs)
    plt.imshow(w)
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}{title}_word_cloud.jpg")


def most_informative_features(classifier,experiment_name, n=100):

    """Plot most informative features"""
    v = classifier[1].coef_.ravel()
    features = classifier[0].get_feature_names_out()
    coefs_with_fns = sorted(zip(v, features))

    top_misc =  {fn_1:coef_1 for coef_1, fn_1 in coefs_with_fns[:-(n+1):-1]}
    top_climate = {fn_1:abs(coef_1) for coef_1, fn_1 in coefs_with_fns[:n:]}

    top_climate_list = list(top_climate.keys())
    top_misc_list = list(top_misc.keys())
    cloud(top_climate,experiment_name+"-"+"climate")
    cloud(top_misc,experiment_name+"-"+"misc")


def main():
    experiment_name = "model_upsampling_aug_1"
    classifier = joblib.load(f"{MODEL_DIR}{experiment_name}")

    # Word cloud plot of the most informative features
    most_informative_features(classifier, experiment_name, 100)


if __name__ == "__main__":
    main()
