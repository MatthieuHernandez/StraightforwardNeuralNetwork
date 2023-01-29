import tensorflow as tf
from keras.layers import *
from Layer import layer_info

# Reshape the weights
weights = tf.reshape(tf.range(1, 5, 1, dtype=tf.float32), (2, 2, 1))

# Add zeros to the input to match SNN
input = tf.range(1, 5, 1, dtype=tf.float32)

# Print info used on the layer used in the C++ unit tests
layer_info(forward_layer=LocallyConnected2D(filters=1, kernel_size=1, strides=1),
           backward_layer=UpSampling2D(size=1),
           weights=weights,
           input_shape=(2, 2, 1),
           input=input,
           error=tf.range(1, 5, 1, dtype=tf.float32))
