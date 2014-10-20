data_dir = 'Tradeshift-Text-Classification/'

## Big Data -- Sample Data!
import pandas as pd
train = pd.read_csv(data_dir + 'train.csv')
train.shape
sample_size = 170000
ratio = train.shape[0] / sample_size

train_sample = train[
    [hash(id) % ratio == 0 for id in train['id']]
]

train_sample.shape

train_sample.to_csv(data_dir + 'train_sample.csv', index = False)

# Free memory

del train

## Try to make something useful
import pandas as pd

data_dir = 'tradeshift/'
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

### Categorical values encoding
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
            print name, len(np.unique(train_with_labels[name]))
        else :
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
        
X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)

X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

print X_numerical.shape, X_sparse.shape, X_test_numerical.shape, X_test_sparse.shape

X_numerical = np.nan_to_num(X_numerical)
X_test_numerical = np.nan_to_num(X_test_numerical)

from sklearn.externals import joblib

joblib.dump(
    (X_numerical, X_sparse, X_test_numerical, X_test_sparse),
    data_dir + 'X.dump',
    compress = 1,
)

## Trying to predict something
### Build two level classifier, first train base level
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
        rf = RandomForestClassifier(n_estimators = 10, n_jobs = 1)
        rf.fit(X_numerical_base, y)
        X_meta.append(rf.predict_proba(X_numerical_meta))
        X_test_meta.append(rf.predict_proba(X_test_numerical))

        svm = LinearSVC()
        svm.fit(X_sparse_base, y)
        X_meta.append(svm.decision_function(X_sparse_meta))
        X_test_meta.append(svm.decision_function(X_test_sparse))
        
X_meta = np.column_stack(X_meta)
X_test_meta = np.column_stack(X_test_meta)

print X_meta.shape, X_test_meta.shape

### Here train meta level and get predictions for test set
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
        rf = RandomForestClassifier(n_estimators=30, n_jobs = 1)
        rf.fit(np.hstack([X_meta, X_numerical_meta]), y)

        predicted = rf.predict_proba(np.hstack([X_test_meta, X_test_numerical]))

        predicted = predicted[:, 1]
        
        rf = RandomForestClassifier(n_estimators=30, n_jobs = 1)
        scores = cross_val_score(rf, np.hstack([X_meta, X_numerical_meta]), y, cv = 4, n_jobs = 1, scoring = log_loss_scorer)

        print i, 'RF log-loss: %.4f ± %.4f, mean = %.6f' %(np.mean(scores), np.std(scores), np.mean(predicted))

    
    p_test.append(
        predicted
    )
    
p_test = np.column_stack(p_test)

### Save predictions
p_test.shape

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

save_predictions('quick_start.csv.gz', test['id'].values, p_test)



























