---
layout: default
title: L1 Regularization
parent: Optimizer
grand_parent: Neural network
nav_order: 2
---

# L1 Regularization
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/l1_regularization.png" att="L1 Regularization" width="320px" class="center"/>
</p>

## Presentation
This optimizer is a layer optimizer that add a value (regularization penalty) to the error of a layer that depends on the average. This regularization corresponds to the average value of weights of each neurons in the layer multiply by a small factor.

## Declaration
This is the function used to declare a L1 Regularization optimizer.
```cpp
OptimizerModel L1Regularization(float value);
```
**Arguments**
 * **value**: The value of the factor. This value must be very low, a good value would be between 1e-2 and 1e-5.

Here is an example of L1 Regularization.
```cpp
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(100, activation::sigmoid, L1Regularization(1e-3f)),
        GruLayer(10),
        FullyConnected(10)
    });
```
