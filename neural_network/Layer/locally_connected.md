---
layout: default
title: Locally Connected
parent: Layers
grand_parent: Neural network
nav_order: 6
---

# Locally Connected layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/locally_connected.gif" att="locally connected layer" width="320px" class="center"/>
</p>

## Presentation
This layer is mainly used to reduce the size of input, the Locally Connected layer makes it easier to extract markers from the images. The Locally Connected layer can be 1D or 2D depending on the previous layer.

## Declaration 
This is the function used to declare a Locally Connected layer.
```cpp
template <class ... TOptimizer>
LocallyConnected(int numberOfLocallyConnected, int sizeOfLocalMatrix, activation activation = activation::sigmoid, TOptimizer ... optimizers);
```
**Arguments**
 * **numberOfLocallyConnected**: The number of output filters. Multiply the number of neurons.
 * **sizeOfLocalMatrix**: The size of the matrix. For a 2D convolution the matrix is a square of length `sizeOfLocalMatrix`.
 * **activation**: The activation function of the neurons of the layer. [See list of activation function]({{site.baseurl}}/layer/activation_functions.html)

 Here is an example of neural networks with 2D input. The Locally Connected layer receives a shape of 28 x 28 x 1 input and outputs a size of 7 x 7 x 2 output.
```cpp
 StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        LocallyConnected(2, 4),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    });
```
[See an example of Locally Connected layer on dataset]({{site.baseurl}}/examples/audio_cats_and_dogs.html)