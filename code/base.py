from __future__ import division
#import os
import re
import sys
import ast
import nltk
import copy
import time
import patsy
import bisect
import pickle
import random
import hyperopt
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix, vstack
from sklearn.svm import SVC, SVR, OneClassSVM
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA, TruncatedSVD
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet, Lars, OrthogonalMatchingPursuit, RANSACRegressor, ARDRegression, SGDClassifier


class Unbuffered(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

''' General functions and classes '''

class preprocess_wrapper:

    def __init__(self, funct):

        self.funct = funct
        self.__len__ = 10

    def __len__(self):
        return self.__len__

    def fit_transform(self, train_df, test_df, y_train):

        if self.funct is not None:
            try:
                self.funct.fit(train_df)
            except TypeError:
                self.funct.fit(train_df, y_train)
            train_df = self.funct.transform(train_df)
            test_df = self.funct.transform(test_df)
            return train_df, test_df, y_train
        else:
            return train_df, test_df, y_train

class outlier_removal:

    def __init__(self, args={'kernel': 'rbf'}, algo=OneClassSVM):

        self.algo = algo(**args)
        self.__len__ = 10

    def __len__(self):
        return self.__len__

    def fit_transform(self, train_df, test_df, y_train):

        self.algo.fit(train_df)
        outliers = self.algo.predict(train_df) == -1
        train_df = pd.DataFrame(train_df).loc[~outliers, :]
        y_train = pd.Series(y_train).loc[~outliers]
        return train_df, test_df, y_train

def find_categorical(df):
    pass

# pass preprocessing steps in a list
def preprocessing(train_df, test_df, y_train, steps):

    # do not apply preproc in categorical vars?
    for step in steps:

        preproc_fn = preprocess_wrapper(step)

        if preproc_fn is not None:

            train_df, test_df, y_train = preproc_fn.fit_transform(train_df, test_df, y_train)

    return train_df, test_df, y_train

# classe em que objetos incluem todos os passos de um framework (pipeline)
class supervised_framework:

    def __init__(self, preproc_steps, algorithm, args):

        self.preproc_steps = preproc_steps
        self.algo = algorithm(**args)

    def fit(self, train_df, test_df, y_train):

        self.X_train, self.X_test, self.y_train = preprocessing(train_df, test_df, y_train, self.preproc_steps)
        self.algo.fit(self.X_train, self.y_train)

    def predict(self, test_df):

        predictions = self.algo.predict(self.X_test)
        try:
            probs = self.algo.predict_proba(self.X_test)
            return {'preds': predictions, 'probs': probs}
        except:
            return {'preds': predictions}

    def fit_predict(self, train_df, test_df, y_train):

        self.fit(train_df, test_df, y_train)
        return self.predict(test_df)

def strat_k_fold_cross_val(k, base_df, y, framework):

    kf = StratifiedKFold(y, n_folds=k, shuffle=True)

    count = 0
    results = {}
    for train_inds, test_inds in kf:

        train, y_train, test, y_test = base_df[train_inds,:], y[train_inds], base_df[test_inds,:], y[test_inds]

        output = framework.fit_predict(train, test, y_train)

        results[count] = {'out': output, 'gtruth': y_test}
        count += 1

    return results

def k_fold_cross_val(k, base_df, y, framework):

    kf = KFold(len(y), n_folds=k, shuffle=True)

    count = 0
    results = {}
    for train_inds, test_inds in kf:

        train, y_train, test, y_test = base_df[train_inds,:], y[train_inds], base_df[test_inds,:], y[test_inds]

        output = framework.fit_predict(train, test, y_train)

        results[count] = {'out': output, 'gtruth': y_test}
        count += 1

    return results

def optimize(experimental_fmwk, space, max_evals, trials):

    global eval_number
    eval_number = 0

    fmin(experimental_fmwk, space, algo=tpe.suggest,
         trials=trials, max_evals=max_evals)

    return get_best(trials)


def ensemble(train_df, test_df, frameworks):
    pass

def one_hot_enc(in_df, col_names, name_prefixes):

    df = copy.deepcopy(in_df)

    for i, col_name in enumerate(col_names):

        binarized_df = pd.get_dummies(
            df[col_name], prefix=name_prefixes[i], prefix_sep='')

        df = pd.concat([df, binarized_df], axis=1)

        df = df.drop([col_name], 1)

    return df

def get_intersection(train_df, test_df):

    cols_intersec = [col for col in train_df.keys() if col in test_df.keys()]

    return train_df.loc[:, cols_intersec], test_df.loc[:, cols_intersec]

def get_best(trials, ind=0):

    best_ind = trials.losses().index(sorted(trials.losses())[ind])
    return trials.trials[best_ind]

def get_rm_cols_bool(candidates):

    rm_dict = {}
    for element in candidates:

        rm_dict[element] = hp.choice('col_bool' + str(element), [True, False])

    return rm_dict

def get_rm_cols_list(rm_dict):

    candidates = []
    for key in rm_dict.keys():

        if rm_dict[key]:
            candidates.append(key)

    return candidates

def get_col_inds(df_keys, key_list):

    inds = []
    for key in key_list:
        inds.append(list(df_keys).index(key))

    return inds

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def one_hot_inv(prefixes, names, df):

    new_df = copy.deepcopy(df)

    for i, p in enumerate(prefixes):

        cols = [k for k in new_df.keys() if k.startswith(p)]
        subset = new_df.loc[:, cols]
        transform = subset.apply(
            lambda row: '+'.join([col for col, b in zip(subset.columns, row) if b]), axis=1)
        new_df[names[i]] = transform
        new_df = new_df.drop(cols, 1)

    return new_df

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                       shape=loader['shape'])

''' Titanic Challenge functions '''

def titanic_remove_unwanted(in_df, mode='train'):

    df = copy.deepcopy(in_df)

    # features to be removed:
    rmv = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
    rmv_tst = ['PassengerId', 'Name', 'Ticket', 'Cabin']

    if mode == 'train':
        y = df['Survived']
        df = df.drop(rmv, 1)
        return df, y
    elif mode == 'test':
        df = df.drop(rmv_tst, 1)
        return df
