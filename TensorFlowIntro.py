# Learning TensorFlow
# https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function

 code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# install packages
install("tensorflow")

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np 
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#explore data

train_images.shape
len(train_labels)

train_labels

test_images.shape
len(test_labels)

#preprocess data
plt.figure() #new figure
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

