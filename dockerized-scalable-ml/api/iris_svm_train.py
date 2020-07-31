# %% [markdown]
# Train model to classify images

# %% load required packages

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# %% load data

iris = datasets.load_iris()
# understand dataset
# print("Iris dataset description: ", iris['DESCR'])

X = iris.data # Features
y = iris.target # Target variable
tgt = iris.target_names
print(tgt)
desc = iris.DESCR
print(desc)

# %% split data
X_train , X_test , y_train, y_test = train_test_split(X, y,test_size=0.2,random_state = 10)

# %% Use SVM classifier
model = SVC(kernel='linear').fit(X_train,y_train)

# %% Calculate test prediction
y_pred = model.predict(X_test)
print("model score: " + str(model.score(X_test, y_test.ravel())))

# %% save model
joblib.dump(model,'model/iris_svm_model.pkl',compress=True)      





# %%
