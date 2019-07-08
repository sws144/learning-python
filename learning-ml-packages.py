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