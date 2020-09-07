---
layout: default
title: Training
parent: Neural network
nav_order: 4
---

# Training

## Declaration 
```cpp
StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(20), FullyConnected(3)});

neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s || 0.9_acc);
neuralNetwork.stopTraining();
```
This is basicly how train a neural network. The `startTraining` method recieve a data and start to train the neural network on a new thread. Conversely the `stopTraining` method stop the training of the neural network and close the opened thread. The `WaitFor` method as indiated by his name wait until the given condition will be reached.
The `WaitFor` method can use one or several conditions with the operator `&&` or `||` are used. Don't mix the both operator.

List of all conditions of the `WaitFor` method:
