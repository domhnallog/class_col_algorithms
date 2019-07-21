import csv

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression

labels = []
titles = []
test_titles = []
test_labels = []

with open('/Users/acc/development/lede/2019/01-logistic-regression/data/vg_test.csv', 'r') as csvfile:
    docs = csv.reader(csvfile)
    for doc in docs:
        if doc[0] != 'Title' and doc[1] != 'Is_Landscape':
            labels.append(1 if doc[1] == 'TRUE' else 0)
            titles.append(doc[0].replace('"', "").lower())

with open('/Users/acc/development/lede/2019/01-logistic-regression/data/vg_training.csv', 'r') as csvfile:
    docs = csv.reader(csvfile)
    for doc in docs:
        if doc[0] != 'Title' and doc[1] != 'Is_Landscape':
            test_labels.append(1 if doc[1] == 'TRUE' else 0)
            test_titles.append(doc[0].replace('"', "").lower())

def extract_unigram(train):
  cv = CountVectorizer(min_df=10, max_df=1.0, token_pattern=r'(?u)\b\w+\b')
  train_data = cv.fit_transform(train)
  return train_data, cv

def classify(training_data, training_labels, test_data, test_labels):
    classifier = LogisticRegression()
    classifier.fit(training_data, training_labels)
    predictions = classifier.predict(test_data)
    print(metrics.accuracy_score(test_labels, predictions))
    print(metrics.precision_score(test_labels, predictions))
    print(metrics.precision_score(test_labels, predictions))
    return metrics.accuracy_score(test_labels, predictions), classifier

training_data, cv = extract_unigram(np.array(titles))
training_data = csr_matrix(training_data, dtype=np.float64)
training_labels = np.array(labels, dtype=np.float64)
train = cv.transform(titles)
test = cv.transform(test_titles)
accuracy, clf = classify(train, training_labels, test, test_labels)
print(accuracy)
