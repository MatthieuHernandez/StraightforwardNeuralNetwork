---
layout: default
title: Input
parent: Layers
grand_parent: Neural network
nav_order: 1
---

# Input

## Presentation
Input Layer must be the first layer of any neural network, it is not really a layer (has no neurons) but it provides to define the shape of input. For example for 32 x 32 color images you can declare input like `Input(32, 32, 3)` to have 2D input but you can also use `Input(1024, 3)` or `Input(3072)` to have 1D input. The shape of input (1D or 2D) determines the shape of the next layer. The shape does not change anything for simple Layers like `FullyConnected` if the number of input stay the same but for filter layers like` Convolution` or `LoccalyConnected` the input layer will have the same number of dimensions as the input.

## Declaration 
This is the function used to declare the size input of neural network.
```cpp
template <typename ... TInt>
LayerModel Input(TInt... sizeOfInput)
```
Here is an example of neural networks with 2D input.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(1, 5),
        FullyConnected(70),
        FullyConnected(10)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/MNIST.html)
<br>
_3D inputs or more are not supported._