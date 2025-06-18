import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def __get_layer_output(layer, new_weights, input, expected, lr):
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
        loss = 0.5 * (output - expected)**2

    # Compute gradients
    grads = tape.gradient(loss, [input, *layer.trainable_variables])

    # Update weights
    for v, g in zip(layer.trainable_variables, grads[1:]):
        v.assign_sub(lr * g)
    return output, grads


def layer_info(forward_layer, weights, input, expected, lr):
    # Do a forward propagation
    forward_output, grads = __get_layer_output(forward_layer, weights, input, expected, lr=lr)

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
    print("========================== new_weights ==========================")
    print(forward_layer.weights[0])
    print("========================== gradiant_weights ==========================")
    ww = tf.reshape(tf.constant(weights, dtype=tf.float32), forward_layer.weights[0].shape)
    gradiant_weights = tf.subtract(forward_layer.weights[0], ww)
    print(gradiant_weights)
    if getattr(forward_layer, "use_bias", False):
        print("============================ new_bias ===========================")
        print(forward_layer.weights[1])
        print("========================== gradiant_bias ==========================")
        bb = tf.reshape(tf.constant([1], dtype=tf.float32), forward_layer.weights[1].shape)
        gradiant_bias = tf.subtract(forward_layer.weights[1], bb)
        print(gradiant_bias)

    print("========================== new_outputs ==========================")
    print(forward_layer(input))
