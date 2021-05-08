---
layout: default
title: Error Multiplier
parent: Optimizer
grand_parent: Neural network
nav_order: 4
---

# Error Multiplier

## Presentation
This optimizer just multiply the error of the layer by a value to increase the learning of this layer. It is mainly a test optimizer, avoid using it.

## Declaration
This is the function used to declare a Error Multiplier optimizer.
```cpp
OptimizerModel ErrorMultiplier(float factor);
```
**Arguments**
 * **factor**: The factor by which the error will be multiplied.

Here is an example of ErrorMultiplier.
```cpp
 StraightforwardNeuralNetwork neuralNetwork({
        Input(32, 32, 3),
        Convolution(2, 3, activation::GELU, ErrorMultiplier(50.0f)),
        FullyConnected(100, activation::sigmoid, Dropout(0.2f)),
        FullyConnected(200, activation::sigmoid, Dropout(0.2f)),
        FullyConnected(10)
        },
        StochasticGradientDescent(0.003f, 0.2f));
```