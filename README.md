# Gmail-spam-classification
Python-Machine Learning 

![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![sklearn](https://img.shields.io/badge/Library-NLTK-orange.svg)

<p align="center">
  <br>
  <img alt="Visual Studio Code" width="1080px" src="https://www.bestproxyreviews.com/wp-content/uploads/2020/07/Bad-Conversion-bot.jpg" />
</p>

## About:

We all face the problem of spams in our inboxes. Email spam, are also called as junk emails, are unsolicited messages sent in bulk by email (spamming).Popular email providers like Gmail use spam classifier programs which can tell whether a given message is spam or not!

Classification is a process of a given set of data into classes, It can be performed on both structured or unstructured data.  For example particular email is spam or not a spam. Natural Language Processing (NLP) is a field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language. Since emails are in some language, computers first convert them to numbers using NLP so that they can understand them and then classify whether particular email is spam or not.

## How It Does:
Extract the text and the target class from the dataset. Extract the features of the test using TF-IDF vectorizer for the Input features.Split the skewed data into shuffled sets using stratified shuffle split in sklearn library. Use standard classifiers to classify the data into spam or ham.

## Prerequisites:

-  `Python`
-  `scikit-learn` / `sklearn`
-  `Pandas`
-  `NumPy`
-  `matplotlib`
-  An environment to work in - something like `Jupyter` or `Spyder`

## Dataset:
The SMS/Email Spam Collection is a set of SMS tagged messages that have been collected for SMS/Email Spam research. It contains one set of SMS messages in English of 5,567 messages, tagged according being ham (legitimate) or spam.

> You can collect raw dataset from [here](https://github.com/harshithvh/Gmail-spam-classification/blob/main/spam.tsv).

The files contain one message per line. Each line is composed by two columns:
- `Class`- contains the label (ham or spam) 
- `Message` - contains the raw text.

## Components:
-  Using TF-IDF for feature extraction of the text data for the messages.
-  Use splits for skewed data(Since the number of ham are far more than the number of spam messages,the data is skewed)
-  Use stratified shuffled split for the split of skewed data.
-  Use different standard classifiers for classification of the SMS.
-  Compare the accuracy of various classifiers using standard classification metrics

## Future Scope:
- Adding this feature in a dynamic website which supports contact-us typo feature.
- Show live user inputs for Ham and Spam  .

