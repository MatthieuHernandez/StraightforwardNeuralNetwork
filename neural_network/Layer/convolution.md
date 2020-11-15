---
layout: default
title: Convolution
parent: Layers
grand_parent: Neural network
nav_order: 6
---

# Convolutional layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/convolution.gif" att="Convolutional layer" width="320px" class="center"/>
</p>

## Presentation
This layer is mainly used for image processing, this makes it easier to extract markers from the images. The Convolutional layer can be 1D or 2D depending on the previous layer.
## Declaration 
This is the function used to declare a convolutional layer.
```cpp
 template <class ... TOptimizer>
LayerModel Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activation activation = activation::ReLU, TOptimizer ... optimizers);
```
**Arguments**
 * **numberOfConvolution**: The number of output filters in the convolution. Multiply the number of neurons.
 * **sizeOfConvolutionMatrix**: The size of the convolution matrix. For a 2D convolution the matrix is a square of length `sizeOfConvolutionMatrix`.
 * **activation**: The activation function of the neurons of the layer. [See list of activation function]({{site.baseurl}}/layer/activation_functions.html)

Here is an example of neural networks with 2D input.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        FullyConnected(150, activation::ReLu),
        FullyConnected(70, activation::tanh),
        FullyConnected(10)
    });
```
[See an example of Convotionnal layer on dataset]({{site.baseurl}}/examples/fashion_mnist.html)