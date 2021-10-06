#On the Terminal run - python -m spacy download en_core_web_lg
#On the Terminal run - python -m spacy download en
#Run code in python - nlp = spacy.load("en_core_web_sm")

import pandas as pd 
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Understanding Data set
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv('spam.csv',encoding='cp1252')
print(data.head())
data = data[['v1','v2']]
data['v1'] = data['v1'].apply(lambda x:0 if x=='ham' else 1)
print(data)

# Text Pre-Processing
def process(x):
    temp = []
    document = nlp(x.lower())
    for i in document:
        if i.is_stop!=True and i.is_punct!= True:
            temp.append(i.lemma_)
        else:
            pass

    return (' '.join(temp))

data['v2'] = data['v2'].apply(lambda x: process(x))
print(data.head())

vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
text_vector = vectorizer.fit_transform(data['v2'].values.tolist())
print(text_vector)

# Splitting Data set
x_train, x_test, y_train, y_test = train_test_split(text_vector.toarray(),data['v1'],test_size=0.2,random_state=20)

# Model Building
modelB = BernoulliNB()
modelB.fit(x_train,y_train)
print(modelB.score(x_train,y_train))

y_predictedB = modelB.predict(x_test)

print(accuracy_score(y_test,y_predictedB))

# Best model is BernoulliNB with 98% Accuracy