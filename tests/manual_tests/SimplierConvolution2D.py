import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = [-1]

expected = tf.cast(tf.reshape([2], shape=(1, 1, 1, 1)), dtype=tf.float32)

input = tf.cast(tf.reshape([-2], shape=(1, 1, 1, 1)), dtype=tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=Conv1D(filters=1, kernel_size=1, bias_initializer="ones"),
           weights=weights,
           input=input,
           expected=expected,
           lr=0.001)
