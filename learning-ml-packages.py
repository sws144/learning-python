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
model.fit =(dataset.data, dataset.target)