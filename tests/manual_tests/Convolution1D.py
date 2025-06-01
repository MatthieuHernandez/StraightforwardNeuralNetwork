import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = tf.range(1, 13, 1, dtype=tf.float32)
weights = tf.reshape(weights, (2, 6))
weights = tf.transpose(weights)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=Conv1D(filters=2, kernel_size=3, bias_initializer="ones"),
           weights=weights,
           input=tf.reshape(tf.range(1, 11, 1, dtype=tf.float32), (1, 5, 2)),
           error=tf.reshape(tf.range(1, 7, 1, dtype=tf.float32), (1, 3, 2)))
