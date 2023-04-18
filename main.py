import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

#read dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#assign attributes to track fake or true
fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop = True)

#data shuffle
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)

#drop unwanted data attributes
data.drop(["date"],axis=1,inplace=True)
data.drop(["title"],axis=1,inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())

#remove punctuation
import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)


#split data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

#apply model
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
#prediction = model.predict(X_test)
#print("accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2)))

pickle.dump(model,open('model.pkl','wb'))








