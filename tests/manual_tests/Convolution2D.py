import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = [1, 10, 1, 10, 2, 11, 2, 11, 3, 12, 3, 12, 4, 13, 4, 13, 5,
           14, 5, 14, 6, 15, 6, 15, 7, 16, 7, 16, 8, 17, 8, 17, 9, 18, 9, 18]

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=Conv2D(filters=2, kernel_size=3, strides=1, padding="valid"),
           backward_layer=Conv2DTranspose(filters=2, kernel_size=3,
                                          strides=1, padding="valid"),
           weights=weights,
           input_shape=(5, 5, 2),
           input=tf.range(1, 51, 1, dtype=tf.float32),
           error=tf.range(1, 19, 1, dtype=tf.float32))
