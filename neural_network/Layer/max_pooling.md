---
layout: default
title: Max pooling
parent: Layers
grand_parent: Neural network
nav_order: 5
---

# Locally Connected layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/max_pooling.png" att="max pooling layer" height="500px" class="center"/>
</p>

## Presentation
This layer contains no neurons and is used to reduce the size of the input. The Max Pooling layer can be 1D or 2D depending on the previous layer.


## Declaration 
This is the function used to declare a Max Pooling layer.
```cpp
template <class ... TOptimizer>
LayerModel MaxPooling(int sizeOfPoolingMatrix)
```
**Arguments**
 * **sizeOfLocalMatrix**: The size of the matrix. For a 2D convolution the matrix is a square of length `sizeOfLocalMatrix`.

 Here is an example of neural networks with 2D input. The Max layer receives a shape of 28 x 28 x 1 input and outputs a size of 14 x 14 x 1 output.
```cpp
 StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        MaxPooling(2),
        FullyConnected(10)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/Wine.html)