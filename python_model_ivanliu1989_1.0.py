# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:31:52 2014

@author: ivan
"""

# Big Data -- Sample Data!
import pandas as pd
data_dir = '/Users/ivan/Work_directory/TTC/data/'
train = pd.read_csv(data_dir + 'train.csv')
train.shape
sample_size = 170000
ratio = train.shape[0] / sample_size

train_sample = train[
    [hash(id) % ratio == 0 for id in train['id']]
]

train_sample.shape
train_sample.to_csv(data_dir + 'train_sample.csv', index = False)
del train

# Try to make something useful
train_sample = pd.read_csv(data_dir + 'train_sample.csv')
labels = pd.read_csv(data_dir + 'trainLabels.csv')
labels.columns
train_with_labels = pd.merge(train_sample, labels, on = 'id')
train_with_labels.shape

from collections import Counter
Counter([name[0] for name in train_with_labels.columns])
del labels
del train_sample
test = pd.read_csv(data_dir + 'test.csv')

# Categorical values encoding
from sklearn.feature_extraction import DictVectorizer

X_numerical = []
X_test_numerical = []
vec = DictVectorizer()
names_categorical = []

train_with_labels.replace('YES', 1, inplace = True)
train_with_labels.replace('NO', 0, inplace = True)
train_with_labels.replace('nan', np.NaN, inplace = True)

test.replace('YES', 1, inplace = True)
test.replace('NO', 0, inplace = True)
test.replace('nan', np.NaN, inplace = True)


for name in train_with_labels.columns :    
    if name.startswith('x') :
        column_type, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x: x[1])
        
        # LOL expression
        if column_type == str(str) :
            train_with_labels[name] = map(str, train_with_labels[name])
            test[name] = map(str, test[name])

            names_categorical.append(name)
            print (name, len(np.unique(train_with_labels[name])))
        else :
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
        
X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)

X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

print X_numerical.shape, X_sparse.shape, X_test_numerical.shape, X_test_sparse.shape