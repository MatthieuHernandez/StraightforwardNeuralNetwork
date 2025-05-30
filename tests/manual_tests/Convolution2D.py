import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = [1, -10, 1, -10, 2, 11, 2, -11, 3, 12, 3, 12, -4, -13, -4, 13, -5,
           14, 5, -14, 6, 15, -6, -15, 7, -16, 7, 7, -8, 2, -8, 1, 9, -8, 6, 1]

expected = tf.cast(tf.reshape([8, 159, 7, 250, -10, 175,
                               -16, 63, 16, 79, -17, 22,
                               0, -140, 21, -90, 37, -3], shape=(1, 3, 3, 2)),
                   dtype=tf.float32)

input = tf.cast(tf.reshape([1, -1, 2, -2, 3, -3, 2, -2, 3, -3,
                            1, -1, 1, -1, 2, -2, 3, -3, 2, -2,
                            -3, 0, -3, 1, -1, 1, -1, 2, -2, 3,
                            -3, 2, -2, 3, -3, 1, -1, 0, 1, -1,
                            2, -2, 3, -3, 2, -2, 3, -3, 1, -1], shape=(1, 5, 5, 2)),
                   dtype=tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=Conv2D(filters=2, kernel_size=3, bias_initializer="ones"),
           weights=weights,
           input=input,
           expected=expected,
           lr=0.00001)
