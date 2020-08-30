---
layout: default
title: Fully Connected
parent: Layers
grand_parent: Neural network
nav_order: 2
---

# Fully Connected layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/fully_connected_layer.png" att="Fully Connected layer" width="200px" class="center"/>
</p>

## Presentation
This layer is the most common and simple layer, almost all neural network have once or more. The fully connected layer is a layer with simple neuron where all neurons is connected to all input. 
## Declaration 
This is the function used to declare a fully connected layer.
```cpp
LayerModel FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid);
```
Here is an example of neural networks with 3 fully connected layer.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        FullyConnected(150, activatation::ReLu),
        FullyConnected(70, activatation::tanh),
        FullyConnected(10)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/Wine.html)