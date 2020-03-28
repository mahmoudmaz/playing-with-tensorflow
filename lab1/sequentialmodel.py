import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Download and import the MIT 6.S191 package
#!pip install mitdeeplearning
#!pip install ipython

import mitdeeplearning as mdl
import numpy as np

n_output_nodes = 3
model = Sequential()

dense_layer = Dense(n_output_nodes, activation='sigmoid')
model.add(dense_layer)

x_input = tf.constant([[1,2.]], shape=(1,2))

model_output = model(x_input).numpy()
print(model_output)