import numpy as np
import tensorflow as tf
from keras.layers import *
print("TensorFlow version:", tf.__version__)


def __get_layer_output(layer, new_weights, bias, input_shape, input):
    # Instanciate a model
    model = tf.keras.models.Sequential([
        Input(input_shape),
        layer,
    ])

    # Retrieve the varaibles
    layer = model.layers[0]
    weights = layer.get_weights()
    batch_shape = (1,) + input_shape

    # Set the weights and bias
    weights[0] = tf.reshape(new_weights, weights[0].shape)
    weights[1] = np.reshape([bias, bias], weights[1].shape)
    model.layers[0].set_weights(weights)

    # Compute the output
    input = tf.reshape(input, batch_shape)
    return layer.call(input), layer.output_shape[1:]


def layer_info(forward_layer,
               backward_layer,
               weights,
               input_shape,
               input,
               error):
    # Do a forward propagation
    forward_output, output_shape = __get_layer_output(
        forward_layer, weights, 1, input_shape, input)

    # Do a backward propagation
    backward_output, _ = __get_layer_output(
        backward_layer, weights, 0, output_shape, error)

    # Print the results
    print("============================= input =============================")
    print(input)
    print("============================ weights ============================")
    print(weights)
    print("============================ output =============================")
    print(forward_output)
    print("============================= error =============================")
    print(backward_output)
