import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = [0.1, 0.0, 0.1, -0.1, 0.2, 0.1, 0.2, -0.1, 0.3, 0.2, 0.3, 0.2, -0.4, -0.3, -0.4, 0.3, -0.1, 0.4,
           0.3, -0.4, 0.6, 0.5, -0.2, -0.5, 0.1, -0.6, 0.1, -0.2, -0.4, 2, -0.8, 1, 0.5, -0.4, 0.3, 0.1]

expected = tf.cast(tf.reshape([0.9, 0.5, 0.7, 0.3, 0.8, 0.5, -0.3, -0.6, 1.0, 0.9,
                                -0.7, 0.2, 0.0, -0.4, 1.0, -0.9, 0.3, -0.1, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.5, 0.7, 0.3,
                                -0.1, 0.7, -0.6, 0.6, 0.6, 0.7, -0.1, 0.3, 0.2, -0.4,
                                0.2, -0.9, 0.3, -0.3, 0.2, 0.2 , 0.0, 0.0, 0.0, -1.0], shape=(1, 5, 5, 2)),
                   dtype=tf.float32)

input = tf.cast(tf.reshape([1, -1, 2, -2, 3, -3, 2, -2, 3, -3,
                            1, -1, 1, -1, 2, -2, 3, -3, 2, -2,
                            -3, 0, -3, 1, -1, 1, -1, 2, -2, 3,
                            -3, 2, -2, 3, -3, 1, -1, 0, 1, -1,
                            2, -2, 3, -3, 2, -2, 3, -3, 1, -1], shape=(1, 5, 5, 2)),
                   dtype=tf.float32)

# To have a symmetrical padding.
input_padded = tf.pad(input, paddings=[[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT', constant_values=0)

# Print info used on the layer used in the C++ unit tests
layer = Conv2D(filters=2, kernel_size=3, padding="valid",
               bias_initializer="ones", activation="tanh")
layer_info(layer, weights, input_padded, expected, lr=0.01)
