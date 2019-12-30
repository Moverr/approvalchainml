#!flask/bin/python
from flask import Flask
from flask import jsonify



import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import re
import string
import random
import json

from nltk.classify import ClassifierI
from statistics import mode

index = 0
classifier_indexes = []
midvote = 0





app = Flask(__name__)

@app.route('/')
def index():
    return jsonify("Hello, World!")

if __name__ == '__main__':
    app.run(debug=True)