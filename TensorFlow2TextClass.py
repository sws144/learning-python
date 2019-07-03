# Text Classification
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

from __future__ import absolute_import, division, print_function

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# check for install
install("tensorflow")
install("numpy")
install("matplotlib")

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

# Add loss function to model
model.compile(
            optimizer = 'adam',
            loss =  'binary_crossentropy', 
            metrics=['acc']
            )

# Create a validation set to finetune model
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 40,
                    batch_size = 512,
                    validation_data = (x_val,y_val),
                    verbose=1
                    )

# Evaluate model
results = model.evaluate(test_data, test_labels)

print(results)

# Graph of accuracy and history over time for training
# import plotting package
import matplotlib.pyplot as plt 

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(acc)+1)

# "bo" is for blue dot
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
# b is for solid blue line
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show(block=False)

plt.clf() #clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show(block=False)