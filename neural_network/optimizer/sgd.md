---
layout: default
title: SGD
parent: Optimizer
grand_parent: Neural network
nav_order: 1
---

# Stochastic Gradient Descent

The SGD + Momentum is currently the only neural network optimizer available. The learning rate is generally à value between 0.1 and 0.01. If the value is too high the neural network may throw `NaN` value and if the value is too low the neural networks will not learn. It is possible to compensate for a low learning rate value by increasing the momentum value. The momentum can be set to 0.99 but must remain strictly less than 1.0 otherwise the error will increase.
The default value of learning rate is 0.03 and 0 for the momentum.

```cpp
neuralNetwork.optimizer.learningRate = 0.03;
neuralNetwork.optimizer.momentum = 0.90;
```

