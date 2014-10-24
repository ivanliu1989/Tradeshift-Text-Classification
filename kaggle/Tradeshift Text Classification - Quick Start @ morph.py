# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# @ morph, for the YSDA ML Trainings 18 October, 2014

# <headingcell level=2>

# Download data

# <rawcell>

# data_dir = 'tradeshift/'
# !mkdir {data_dir}

# <rawcell>

# !wget 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3984/train.csv.gz?sv=2012-02-12&se=2014-10-21T00%3A06%3A50Z&sr=b&sp=r&sig=cupgPW%2BU6BpdsnrykcEBBRqLEW565pXYQ6k%2FSc0Me1M%3D' -O {data_dir + 'train.csv.gz'}

# <rawcell>

# !wget 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3984/test.csv.gz?sv=2012-02-12&se=2014-10-21T00%3A09%3A52Z&sr=b&sp=r&sig=YLQCFyAdhIRnz2o4p24zRssUjHYjQ1xOHuTKFsdLxu8%3D' -O {data_dir + 'test.csv.gz'}

# <rawcell>

# !wget 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3984/trainLabels.csv.gz?sv=2012-02-12&se=2014-10-21T00%3A11%3A04Z&sr=b&sp=r&sig=%2Bm9sbZYXOY8L80d1PJEdumGPXvkQby2rpkVOf1fvjUM%3D' -O {data_dir + 'trainLabels.csv.gz'}

# <headingcell level=2>

# Unpack

# <rawcell>

# %%time
# 
# !gunzip {data_dir + '*.gz'}

# <rawcell>

# !ls -l -h {data_dir}

# <headingcell level=3>

# Big Data -- Sample Data!

# <codecell>

import pandas as pd
data_dir='../data/'

# <codecell>

train = pd.read_csv(data_dir + 'train.csv')

# <codecell>

train.shape

# <codecell>

sample_size = 170000
ratio = train.shape[0] / sample_size

train_sample = train[
    [hash(id) % ratio == 0 for id in train['id']]
]

train_sample.shape

# <codecell>

train_sample.to_csv(data_dir + 'train_sample.csv', index = False)

# <codecell>

# Free memory

del train

# <headingcell level=2>

# Try to make something useful

# <codecell>

import pandas as pd

data_dir = '../data/'

# <codecell>

train_sample = pd.read_csv(data_dir + 'train_sample.csv')

# <codecell>

labels = pd.read_csv(data_dir + 'trainLabels.csv')

# <codecell>

labels.columns

# <codecell>

train_with_labels = pd.merge(train_sample, labels, on = 'id')

# <codecell>

train_with_labels.shape

# <codecell>

from collections import Counter

Counter([name[0] for name in train_with_labels.columns])

# <codecell>

del labels
del train_sample

# <codecell>

test = pd.read_csv(data_dir + 'test.csv')

# <headingcell level=3>

# Categorical values encoding

# <codecell>

from sklearn.feature_extraction import DictVectorizer
import numpy as np

X_numerical = []
X_test_numerical = []

vec = DictVectorizer()

names_categorical = []

train_with_labels.replace('YES', 1, inplace = True)
train_with_labels.replace('NO', 0, inplace = True)
#train_with_labels.replace('nan', np.NaN, inplace = True)

test.replace('YES', 1, inplace = True)
test.replace('NO', 0, inplace = True)
#test.replace('nan', np.NaN, inplace = True)


for name in train_with_labels.columns :    
    if name.startswith('x') :
        column_type, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x: x[1])
        
        # LOL expression
        if column_type == str(str) :
            train_with_labels[name] = map(str, train_with_labels[name])
            test[name] = map(str, test[name])

            names_categorical.append(name)
            print name, len(np.unique(train_with_labels[name]))
        else :
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
        
X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)

X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

print X_numerical.shape, X_sparse.shape, X_test_numerical.shape, X_test_sparse.shape

# <codecell>

#X_numerical = np.nan_to_num(X_numerical)
#X_test_numerical = np.nan_to_num(X_test_numerical)

# <codecell>

from sklearn.externals import joblib

joblib.dump(
    (X_numerical, X_sparse, X_test_numerical, X_test_sparse),
    data_dir + 'X.dump',
    compress = 1,
)

# <headingcell level=2>

# Trying to predict something

# <headingcell level=3>

# Build two level classifier, first train base level

# <codecell>

from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

log_loss_scorer = make_scorer(log_loss, needs_proba = True)

y_columns = [name for name in train_with_labels.columns if name.startswith('y')]

X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(
    X_numerical, 
    X_sparse, 
    train_with_labels[y_columns].values,
    test_size = 0.5
)

X_meta = [] 
X_test_meta = []

print "Build meta"

for i in range(y_base.shape[1]) :
    print i
    
    y = y_base[:, i]
    if len(np.unique(y)) == 2 : 
        rf = RandomForestClassifier(n_estimators = 10, n_jobs = 16)
        rf.fit(X_numerical_base, y)
        X_meta.append(rf.predict_proba(X_numerical_meta))
        X_test_meta.append(rf.predict_proba(X_test_numerical))

        svm = LinearSVC()
        svm.fit(X_sparse_base, y)
        X_meta.append(svm.decision_function(X_sparse_meta))
        X_test_meta.append(svm.decision_function(X_test_sparse))
        
X_meta = np.column_stack(X_meta)
X_test_meta = np.column_stack(X_test_meta)

# <codecell>

print X_meta.shape, X_test_meta.shape

# <headingcell level=3>

# Here train meta level and get predictions for test set

# <codecell>

p_test = []

for i in range(y_base.shape[1]) :
    y = y_meta[:, i]

    constant = Counter(y)
    constant = constant[0] < 4 or constant[1] < 4
    
    predicted = None
    
    if constant :
        # Best constant
        constant_pred = np.mean(list(y_base[:, i]) + list(y_meta[:, i]))
        
        predicted = np.ones(X_test_meta.shape[0]) * constant_pred
        print "%d is constant like: %f" % (i, constant_pred)
    else :
        rf = RandomForestClassifier(n_estimators=30, n_jobs = 16)
        rf.fit(np.hstack([X_meta, X_numerical_meta]), y)

        predicted = rf.predict_proba(np.hstack([X_test_meta, X_test_numerical]))

        predicted = predicted[:, 1]
        
        rf = RandomForestClassifier(n_estimators=30, n_jobs = 16)
        scores = cross_val_score(rf, np.hstack([X_meta, X_numerical_meta]), y, cv = 4, n_jobs = 1, scoring = log_loss_scorer)

        print i, 'RF log-loss: %.4f ± %.4f, mean = %.6f' %(np.mean(scores), np.std(scores), np.mean(predicted))

    
    p_test.append(
        predicted
    )
    
p_test = np.column_stack(p_test)

# <headingcell level=3>

# Save predictions

# <codecell>

p_test.shape

# <codecell>

import gzip

def save_predictions(name, ids, predictions) :
    out = gzip.open(name, 'w')
    print >>out, 'id_label,pred'
    for id, id_predictions in zip(test['id'], p_test) :
        for y_id, pred in enumerate(id_predictions) :
            if pred == 0 or pred == 1 :
                pred = str(int(pred))
            else :
                pred = '%.6f' % pred
            print >>out, '%d_y%d,%s' % (id, y_id + 1, pred)

# <codecell>

save_predictions('quick_start.csv.gz', test['id'].values, p_test)

# <codecell>

!ls -l -h  quick_start*.csv.gz

# <headingcell level=3>

# Public result

# <headingcell level=3>

# Quick start on 10% of train - 0.0212323

# <codecell>


