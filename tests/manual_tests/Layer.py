import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def __get_layer_output(layer, new_weights, input, expected):
    # Build the layer
    layer.build(input.shape)

    # Retrieve the variables
    weights = layer.get_weights()

    # Set the weights and bias
    if weights:
        weights[0] = tf.reshape(new_weights, weights[0].shape)
        layer.set_weights(weights)

    # Record gradients
    with tf.GradientTape() as tape:
        tape.watch(input)
        # Compute the output and the loss.
        output = layer(input)
        loss = (output - expected)**2

    # Compute gradients
    grads = tape.gradient(loss, [input, *layer.trainable_variables])
    return output, grads


def layer_info(forward_layer, weights, input, expected):
    # Do a forward propagation
    forward_output, grads = __get_layer_output(forward_layer, weights, input, expected)

    # Print the results
    print("============================= input =============================")
    print(input)
    print("============================ weights ============================")
    print(weights)
    print("============================ output =============================")
    print(forward_output)
    print("============================= error =============================")
    backward_output = grads[0]
    print(backward_output)
