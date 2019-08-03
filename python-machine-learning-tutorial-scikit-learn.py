# https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

# run from command line if missing
# pip install "sklearn" # for example

# %% import packages
import sklearn;
print(sklearn.__version__)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# random forest models
from sklearn.ensemble import RandomForestRegressor

# cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# evaluation metrics
from sklearn.metrics import mean_squared_error,r2_score

# persist model for future use
import joblib #used to be from sklearn

# %% load data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)

print(data.head()) # shows issue bc csv uses ;

data = pd.read_csv(dataset_url, ";")

print(data.head()) # fixed

# understand data
print(data.shape)
print(data.describe())

y = data.quality
x = data.drop('quality',axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2,
                                                    random_state=123,
                                                    stratify=y)

# %% Need to transform data for fitting so have mean 0 & std 1

# lazy way of scaling data (not used)
X_train_scaled_lazy = preprocessing.scale(X_train)
print(X_train_scaled_lazy)
# check dataset scaling
print(X_train_scaled_lazy.mean(axis=0))
print(X_train_scaled_lazy.std(axis=0))

# instead, use transformer API
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

# applying transformer
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))
# should be not mean 0, std 1, as using scaling from train

# but in process, don't even need above, just need to use the scaling object
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# %% step 6 tune hyper parameters
# hyper parameters cannot be trained by model itself
# for random forest, can use mean squared error or mean abs error
print(pipeline.get_params())

# python dictionary
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5,3,1]}

# %% 7 tune model using cross validation
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# fit & tune model
clf.fit(X_train,Y_train)

# see best params
print(clf.best_params_)

# %% 8 refit on training set
# is done by default by GridSearch
print(clf.refit)

# %% 9 eval model pipeline on test data
Y_pred = clf.predict(X_test)

print(r2_score(Y_test,Y_pred))
# 0.4686...

print(mean_squared_error(Y_test,Y_pred))
# 0.35

# but is model good enough?

# %% 10 export model
joblib.dump(clf, 'rf_regressor.pkl')

#repull, can type %reset in ipython interpreter
import joblib
clf2 = joblib.load('rf_regressor.pkl')

#predict
clf2.predict(X_test)
