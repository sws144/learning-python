# https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

# run from command line if missing
# pip install "sklearn" # for example

# import packages
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