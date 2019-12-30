#!flask/bin/python
from flask import Flask, render_template, url_for, json, jsonify


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

import os


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


def showjson():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static/data", "data.json")
    data = json.load(open(json_url))
    return data


@app.route('/')
def index():
    #   this is where the magic is begininig
    try:

        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        json_url = os.path.join(SITE_ROOT, "static", "data.json")
        data = json.load(open(json_url))

        # data ={}
        # with open(data_file) as f:
        #     data = json.load(f)

        print(data['False'][0])

        positive_data = data['True']
        negative_data = data['False']

        positive_data_tokens = []
        negative_data_tokens = []

        for sentence in positive_data:
            # todo:  tockenize this then append
            positive_data_tokens.append(nltk.word_tokenize(sentence))

        for sentence in negative_data:
            # todo:  tockenize this then append
            negative_data_tokens.append(nltk.word_tokenize(sentence))

        stop_words = stopwords.words('english')

        # positive_data_tokens = twitter_samples.tokenized('positive_data.json')
        # negative_data_tokens = twitter_samples.tokenized('negative_data.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_data_tokens:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_data_tokens:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        all_pos_words = get_all_words(positive_cleaned_tokens_list)

        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(10))

        positive_tokens_for_model = get_tweets_for_model(
            positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(
            negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        print(len(dataset))

        train_data = dataset[:5]
        test_data = dataset[5:]

        names = ["MultinomialNBclassifier", "BernoulliNB", "LogisticRegression_classifier", "SGDClassifier_classifier ",
                "SVC_classifier", "LinearSVC_classifier", "NaiveBayesClassifier"]

        MultinomialNBclassifier = SklearnClassifier(MultinomialNB())
        MultinomialNBclassifier.train(train_data)
        print("\nMultinomialNB Accuracy is:",
            (classify.accuracy(MultinomialNBclassifier, test_data)) * 100)

        # GaussianNBclassifier = SklearnClassifier(GaussianNB())
        # GaussianNBclassifier.train(train_data)
        # print("\nGaussianNB Accuracy is:", classify.accuracy(GaussianNBclassifier, test_data))

        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(train_data)
        print("BernoulliNB_classifier Algo Accuracy: ",
            (nltk.classify.accuracy(BernoulliNB_classifier, test_data)) * 100)

        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(train_data)
        print("LogisticRegression Algo Accuracy: ",      (nltk.classify.accuracy(
            LogisticRegression_classifier, test_data)) * 100)

        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(train_data)
        print("SGDClassifier Algo Accuracy: ",
            (nltk.classify.accuracy(SGDClassifier_classifier, test_data)) * 100)

        SVC_classifier = SklearnClassifier(SVC())
        SVC_classifier.train(train_data)
        print("SVC Algo Accuracy: ",
            (nltk.classify.accuracy(SVC_classifier, test_data)) * 100)

        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(train_data)
        print("LinearSVC Algo Accuracy: ",
            (nltk.classify.accuracy(LinearSVC_classifier, test_data)) * 100)

        NuSVC_classifier = SklearnClassifier(NuSVC(gamma='auto'))

        NuSVC_classifier.train(train_data)
        print("NuSVC Algo Accuracy: ",
            (nltk.classify.accuracy(NuSVC_classifier, test_data)) * 100)

        classifier = NaiveBayesClassifier.train(train_data)

        print("Accuracy is:", classify.accuracy(classifier, test_data))

        print(classifier.show_most_informative_features(10))

        custom_tweet = " Bobiwine is contesting for presidency in 2021"

        custom_tokens = remove_noise(word_tokenize(custom_tweet))

        print(custom_tweet, classifier.classify(
            dict([token, True] for token in custom_tokens)))

        voted_classifier = VoteClassifier(classifier, MultinomialNBclassifier, BernoulliNB_classifier,
                                        LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)

        print("Voted Classifier Algo Accuracy: ",
            (nltk.classify.accuracy(voted_classifier, test_data)) * 100)
    except:
        print("An exception occurred")


if __name__ == '__main__':
    app.run(debug=False)
