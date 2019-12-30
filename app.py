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


class VoteClassifier(ClassifierI):
    # List of classifiers passsed to this
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        index = 0
        for c in self._classifiers:
            v = c.classify(features)
            index = index + 1
            votes.append(v)
            classifier_indexes.append(index)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

    def predict(self, text):
        pass


def remove_noise(tweet_tokens, stop_words=()):
    lemmetizer = WordNetLemmatizer()
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            # // we are chainge words back to the original flow
            lametizedword = lemmetizer.lemmatize(token.lower())
            cleaned_tokens.append(lametizedword)
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)







app = Flask(__name__)

@app.route('/')
def index():
    return jsonify("Hello, World!")

if __name__ == '__main__':
    app.run(debug=True)