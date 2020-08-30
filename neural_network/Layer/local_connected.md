---
layout: default
title: Locally Connected
parent: Layers
grand_parent: Neural network
nav_order: 5
---

# Locally Connected layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/locally_connected.gif" att="Convolutionnal layer" width="320px" class="center"/>
</p>

## Presentation
This layer is mainly used for reduce the size of input, the Locally Connected layer makes it easier to extract markers from the images. 
## Declaration 
This is the function used to declare the size input of neural network.
```cpp
LayerModel FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid);
```
Here is an example of neural networks with 2D input.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        FullyConnected(150, activatation::ReLu),
        FullyConnected(70, activatation::tanh),
        FullyConnected(10)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/Wine.html)