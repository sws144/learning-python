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
print("Training entries: {}, labels: {}".format(len(train_data),len(train_labels)))
print(train_data[0])

len(train_data[0]),len(train_data[1])

# Convert integers back to words 
word_index = imdb.get_word_index()

# The first indices
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

# show first review in words
decode_review(train_data[0])

# Prepare data
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding = 'post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding = 'post',
                                                        maxlen=256)

len(train_data[0]), len(train_data[1])
print(train_data[0])

# Build the model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()
