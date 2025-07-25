import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Reshape the weights
weights = tf.range(1, 145, 1, dtype=tf.float32)
weights = tf.reshape(weights, (2, 18, 4))

# Add zeros to the input to match SNN
input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 0,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 0, 0,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], shape=(6, 6, 2), dtype=tf.float32)
input = tf.reshape(input, (1, 6, 6, 2))
expected = tf.cast(tf.reshape([5812, 6213, 12452, 12123, 39110, 39123, 41320, 41642],
                              shape=(1, 2, 2, 2)), tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=LocallyConnected2D(filters=2, kernel_size=3, strides=3,
                                            bias_initializer="ones"),
           weights=weights,
           input=input,
           expected=expected)
