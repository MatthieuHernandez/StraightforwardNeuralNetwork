import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Reshape the weights
weights = tf.range(1, 25, 1, dtype=tf.float32)
weights = tf.reshape(weights, (2, 6, 2))

# Add zeros to the input to match SNN
input_raw = tf.range(1, 11, 1, dtype=tf.float32)
zeros = tf.zeros((2), dtype=tf.float32)
input = tf.concat([input_raw, zeros], 0)
input = tf.reshape(input, (1, 6, 2))
expected = tf.cast(tf.reshape([160, 182, 552, 580], shape=(1, 2, 2)), tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=LocallyConnected1D(filters=2, kernel_size=3, strides=3,
                                            bias_initializer="ones"),
           weights=weights,
           input=input,
           expected=expected)
