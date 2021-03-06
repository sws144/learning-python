# https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("numpy")

# NumPy is a very popular python library for large multi-dimensional array and matrix processing, 
# with the help of a large collection of high-level mathematical functions. 
# It is very useful for fundamental scientific computations in Machine Learning. 
# It is particularly useful for linear algebra, Fourier transform, and random number capabilities. 
# High-end libraries like TensorFlow uses NumPy internally for manipulation of Tensors.

import numpy as np

# creating two arrays of rank 2
x = np.array([[1,2],[3,4]])
y = np.array([[5,6], [7,8]])

# two arrays of rank 1
v = np.array([9,10])
w = np.array([11,12])

#inner producct
print(np.dot(v,w), "\n")

# matrix and vector product
print(np.dot(x,v), "\n")

# matrix and matrix product
print(np.dot(x,y))


# Scipy for mathematics/optimization/statistics, incl. image manipulation
# consider using skimage later
install("scipy")
install("imageio")
install("visvis")

from scipy.misc import imread, imsave, imresize 
import imageio
import visvis as vv


# Read a JPEG image into a numpy array
img = imageio.imread('C:\Stuff\Important\CareerNCollege\Ad Hoc\Git\learning-python\\fruit.jpg')

# print image
vv.imshow(img)

# Tinting the image (using R G B notation)
img_tint = img * [1, 0.45, 0.3] 

# Saving the tinted image 
imageio.imwrite('C:\Stuff\Important\CareerNCollege\Ad Hoc\Git\learning-python\\fruit_tinted.jpg', img_tint) 

# print tinted image
vv.imshow(img_tint)

# Resizing the tinted image to be 300 x 300 pixels 
img_tint_resize = imresize(img_tint, (300, 300)) 

# Saving the resized tinted image 
imageio.imwrite('C:\Stuff\Important\CareerNCollege\Ad Hoc\Git\learning-python\\fruit_tinted_rezised.jpg', img_tint)
vv.imshow(img_tint_resize, 2) 


# scikit-learn
# classical ml algorithms 

install("sklearn")

# sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load iris datasets
dataset = datasets.load_iris()

# fit a CART model to data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Theano #
# mathematical expressions for large datasets

install("theano")

import theano 
import theano.tensor as T
x = T.dmatrix('x')
s = 1 / (1+T.exp(-x))
logistic = theano.function([x],s)
logistic([[0,1],[-1,-2]])


# Tensorflow
# Google-based high performance computing

install("tensorflow")
import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply 
result = tf.multiply(x1,x2)

# Initialize session
sess = tf.Session()

print(sess.run(result))

# Close session
sess.close()

# Keras
# ML library on top of others

# Pytorch
# Computer vision or NLP
# Example for 2 layer network 

# see https://pytorch.org/get-started/locally/#anaconda-1
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") Uncomment to run on GPU

## TODO later


# Pandas 
# for data analysis

install("pandas")
import pandas as pd 

# to make it dictionary
data = {"country": ["Brazil", "Russia", "India", "China", "South Africa"], 
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"], 
       "area": [8.516, 17.10, 3.286, 9.597, 1.221], 
       "population": [200.4, 143.5, 1252, 1357, 52.98] } 

data_table = pd.DataFrame(data)  # to make it into pandas dataframe

print(data_table)

# Matplotlib
# for linear plot 

install("matplotlib")
install("numpy")

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)

plt.plot(x,x,label='linear')
plt.show(block = False) #show without stopping code
plt.legend()