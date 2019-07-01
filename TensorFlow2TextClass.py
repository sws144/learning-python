# Text Classification
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

from __future__ import absolute_import, division, print_function

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# import relevant packages
import tensorflow as tf 
from tensorflow import keras 

import numpy as np

print("tensorflow version: " + tf.__version__)

# Download dataset
imdb = keras.datasets.imdb 
(train_data, train_labels) , (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data
