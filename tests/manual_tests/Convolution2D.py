import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = [1, 10, 1, 10, 2, 11, 2, 11, 3, 12, 3, 12, 4, 13, 4, 13, 5,
           14, 5, 14, 6, 15, 6, 15, 7, 16, 7, 16, 8, 17, 8, 17, 9, 18, 9, 18]
expected = tf.cast(tf.reshape([1602, 3780, 1732, 4212, 1911, 4722,
                               2508, 6377, 2686, 6855, 2844, 7333,
                               3409, 8800, 3581, 9332, 3712, 9812], shape=(1, 3, 3, 2)),
                   dtype=tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=Conv2D(filters=2, kernel_size=3, bias_initializer="ones"),
           weights=weights,
           input=tf.reshape(tf.range(1, 51, 1, dtype=tf.float32), (1, 5, 5, 2)),
           expected=expected)
