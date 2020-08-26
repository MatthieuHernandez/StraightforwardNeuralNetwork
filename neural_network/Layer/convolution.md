---
layout: default
title: Convolution
parent: Layers
grand_parent: Neural network
nav_order: 6
---

# Convolutial layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/convolution.gif" att="Convolutionnal layer" width="320px" class="center"/>
</p>

## Presentation
This layer is mainly used for image processing, this makes it easier to extract markers from the images. The Convolutionnal layer can be 1D ou 2D depending on the previous layer.
## Declaration 
This is the function used to declare the size input of neural network.
```cpp
LayerModel Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activation activation = activation::ReLU);
```
**Arguments**
 * **numberOfConvolution**: the number of output filters in the convolution. [See list of activation function]

 * **activation**: Activation function of neurons. [See list of activation function]({{site.baseurl}}/layer/activation_functions.html)

Here is an example of neural networks with 2D input.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        FullyConnected(150, activatation::ReLu),
        FullyConnected(70, activatation::tanh),
        FullyConnected(10)
    });
```
[See an example of Convotionnal layer on dataset]({{site.baseurl}}/examples/fashion_mnist.html)