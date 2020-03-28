import tensorflow as tf 

import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np

import random
from tqdm import tqdm

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)

test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)


def build_fc_model():
  fc_model = tf.keras.Sequential([
      # First define a Flatten layer
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation='softmax'),

  ])
  return fc_model

model = build_fc_model()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 64
EPOCHS = 5
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss)
print(test_acc)