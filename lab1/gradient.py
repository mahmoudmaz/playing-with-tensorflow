
### Defining a model using subclassing and specifying custom behavior ###
import tensorflow as tf
# Download and import the MIT 6.S191 package
#!pip install mitdeeplearning
#!pip install ipython
import sys

import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


x = tf.Variable(3.)

with tf.GradientTape() as tape:
    y = x * 2

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())

print("-------")

x = tf.Variable([tf.random.normal([1])])
print("Initializing x={}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

for i in range(500):
  with tf.GradientTape() as tape:
    '''TODO: define the loss as described above'''
    loss = (x - x_f)**2

  grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
  new_x = x - learning_rate*grad # sgd update
  x.assign(new_x) # update the value of x
  history.append(x.numpy()[0])


# Plot the evolution of x as we optimize towards x_f!
plt.plot(history)
plt.plot([0, 500],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.savefig("out.png")

