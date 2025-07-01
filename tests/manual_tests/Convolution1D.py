import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Transpose the weights to match SNN
weights = tf.range(1, 13, 1, dtype=tf.float32)
weights = tf.reshape(weights, (2, 6))
weights = tf.transpose(weights)
expected = tf.cast(tf.reshape([50., 110., 93., 219., 134., 330., 178., 445., 92., 294.],
                              shape=(1, 5, 2)), tf.float32)

input = tf.reshape(tf.range(1, 11, 1, dtype=tf.float32), (1, 5, 2))

# To have a symmetrical padding.
input_padded = tf.pad(input, paddings=[[0,0], [1,1], [0,0]], mode='CONSTANT', constant_values=0)

# Print info used on the layer used in the C++ unit tests
layer = Conv1D(filters=2, kernel_size=3, padding="valid",
               bias_initializer="ones", activation=None)
layer_info(layer, weights, input_padded, expected, lr=0.01)
