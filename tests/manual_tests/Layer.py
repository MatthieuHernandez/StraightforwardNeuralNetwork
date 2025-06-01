import numpy as np
import tensorflow as tf
from keras.layers import *

print("TensorFlow version:", tf.__version__)


def __get_layer_output(layer, new_weights, bias, input):
    # Build the layer
    layer.build(input.shape)

    # Retrieve the variables
    weights = layer.get_weights()

    # Set the weights and bias
    if weights:
        weights[0] = tf.reshape(new_weights, weights[0].shape)
        new_bias = tf.ones_like(weights[1]) * bias
        weights[1] = np.reshape(new_bias, weights[1].shape)
        layer.set_weights(weights)

    # Compute the output
    output = layer(input)
    if not weights:
        if tf.math.reduce_prod(output.shape) < tf.math.reduce_prod(new_weights.shape):
            output = tf.concat([output, output], axis=-1)
        output = tf.reshape(output, new_weights.shape) * new_weights
        output = tf.math.reduce_sum(output, axis=0)
    return output


def layer_info(forward_layer, backward_layer, weights, input, error):
    # Do a forward propagation
    forward_output = __get_layer_output(forward_layer, weights, 1, input)

    # Do a backward propagation
    backward_output = __get_layer_output(backward_layer, weights, 0, error)

    # Print the results
    print("============================= input =============================")
    print(input)
    print("============================ weights ============================")
    print(weights)
    print("============================ output =============================")
    print(forward_output)
    print("============================= error =============================")
    print(backward_output)
