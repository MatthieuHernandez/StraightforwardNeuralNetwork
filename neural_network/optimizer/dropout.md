---
layout: default
title: Dropout
parent: Optimizer
grand_parent: Neural network
nav_order: 1
---

# Dropout
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/dropout.png" att="Dropout" width="320px" class="center"/>
</p>

## Presentation
This optimizer is a layer optimizer that randomly disable a certain percentage of neurons during the learning phase.
<br/>
**&#9888;** _No test from this project proves the effectiveness of Dropout._

## Declaration
This is the function used to declare a GRU layer.
```cpp
OptimizerModel Dropout(float value);
```
**Arguments**
 * **value**: The percentage of neurons to randomly disable.

Here is an example of Dropout.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        LocallyConnected(1, 7),
        FullyConnected(150, activation::sigmoid, Dropout(0.4f)),
        FullyConnected(70, Dropout(0.4f)),
        FullyConnected(10)
    });
```

## Algorithms and References

The implementation of the Dropout is based on this paper :
 * [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)