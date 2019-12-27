#!/usr/bin/env python
# coding: utf-8

# In[342]:


import pandas as pd
import numpy as np


# In[480]:


def metrics(actual, predicted):
    """metrics
    Return Acc, precision, recall, f-measure
    """
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            a += 1
        elif actual[i] == 1 and predicted[i] == 0:
            b += 1
        elif actual[i] == 0 and predicted[i] == 1:
            c += 1
        elif actual[i] == 0 and predicted[i] == 0:
            d += 1
    def acc(a, b, c, d):
            return (a + d) * 1.0 / (a + b + c + d)
 
    def p(a, c):
        return a * 1.0 / (a + c)
    
    def r(a, b):
        return a * 1.0 / (a + b)
    
    def f(a, b, c):
        return 2 * a * 1.0 / (2 * a + b + c)

    return acc(a, b, c, d), p(a, c), r(a, b), f(a, b, c)


# In[481]:


import math
def cacl_distance(x1, x2):
    """Cacl the euclidean_distance"""
    dist = 0
    for i in range(len(x1)):
        dist += pow((x1[i] - x2[i]), 2)
    return math.sqrt(dist)


# In[482]:


class KNN():
    def __init__(self, k=5):
        # Init the k in KNN algorithm
        self.k = k

    def predict(self, X_test, X_train, y_train):
        """Input the train_features, train_labels, test_features
        Output the test_labels
        """
        y_predict = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            # The euclidean_distance between the test features and other train features
            distances = np.zeros((X_train.shape[0], 2))
            for j in range(X_train.shape[0]):
                # Cacl the euclidean_distance between the test features and other train features
                dis = cacl_distance(X_test[i], X_train[j])
                label = y_train[j] # The label of train_set
                distances[j] = [dis, label]

                # Get top k object ([dis, label]) sort by ASC
                k_nearest_neighbors = distances[distances[:, 0].argsort()][:self.k]

                # Count the times of each labels appears in K neighbors
                counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))

                # Given the label which appears most in counts
                testLabel = counts.argmax()
                y_predict[i] = testLabel

        return y_predict


# In[505]:


def load_data(file_name):
    """Load data as pandas dataframe and convert to Numpy ndarray
    split the data to train and test set
    """
    data = pd.read_csv(file_name, delimiter='\t', header=-1)
    # Split columns
    x = data.iloc[:, :-1].values
    y = [i[0] for i in data.iloc[:, -1:].values]
    return x, y


# In[506]:


def data_split(X, y, test_size=0.3):
    """Split the data into train and test sets"""
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


# ### 1. project3_dataset1

# In[517]:


x, y = load_data('project3_dataset1.txt')
X_train, X_test, y_train, y_test = data_split(x, y, test_size=0.3)


# In[508]:


model = KNN(k=3)
y_pred = model.predict(X_test, X_train, y_train)
a, p, r, f = metrics(y_test, y_pred)
print "Accuracy: %s\nPrecision: %s\nRecall: %s\nF-measure: %s\n" % (a, p, r, f)


# ### 2. project3_dataset2

# In[513]:


def encode_string(dataset, col):
    m = dict()
    total = 0
    for i in range(len(dataset)):
        if dataset[i][col] not in m:
            m[dataset[i][col]] = total
            total += 1
        dataset[i][col] = m[dataset[i][col]]


# In[514]:


x, y = load_data('project3_dataset2.txt')
encode_string(x, 4)
X_train, X_test, y_train, y_test = data_split(x, y, test_size=0.3)


# In[515]:


model = KNN(k=3)
y_pred = model.predict(X_test, X_train, y_train)
a, p, r, f = metrics(y_test, y_pred)
print "Accuracy: %s\nPrecision: %s\nRecall: %s\nF-measure: %s\n" % (a, p, r, f)


# ### 3. project3_dataset3

# In[466]:


import pandas as pd
train_set = pd.read_csv('project3_dataset3_train.txt', delimiter='	', header=-1)
X_train = train_set.iloc[:, :-1].values
y_train = [i[0] for i in train_set.iloc[:, -1:].values]

test_set = pd.read_csv('project3_dataset3_test.txt', delimiter='	', header=-1)
X_test = test_set.iloc[:, :-1].values
y_test = [i[0] for i in test_set.iloc[:, -1:].values]


# In[472]:


model = KNN(k=3)
y_pred = model.predict(X_test, X_train, y_train)
a, p, r, f = metrics(y_test, y_pred)
print "Accuracy: %s\nPrecision: %s\nRecall: %s\nF-measure: %s\n" % (a, p, r, f)


# In[ ]:




