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
