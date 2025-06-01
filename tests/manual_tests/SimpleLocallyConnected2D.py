import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Reshape the weights
weights = tf.reshape(tf.range(1, 5, 1, dtype=tf.float32), (2, 2, 1))

# Add zeros to the input to match SNN
input = tf.reshape(tf.range(1, 5, 1, dtype=tf.float32), (1, 2, 2, 1))

expected = tf.cast(tf.reshape([3, 6, 9, 16], shape=(1, 2, 2, 1)), tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=LocallyConnected2D(filters=1, kernel_size=1, strides=1,
                                            bias_initializer="ones"),
           weights=weights,
           input=input,
           expected=expected)
