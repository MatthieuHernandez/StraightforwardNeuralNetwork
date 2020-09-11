---
layout: default
title: Training
parent: Neural network
nav_order: 4
---

# Training &#127947;
This part is about how traing a `StraightforwardNeuralNetwork`.

## Example 
```cpp
StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(20), FullyConnected(3)});
neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s || 0.9_acc);
neuralNetwork.stopTraining();
```
This is basicly how train a neural network. The `startTraining` method recieve a data and start to train the neural network on a new thread. Conversely the `stopTraining` method stop the training of the neural network and close the opened thread. The `WaitFor` method as indiated by his name wait until the given condition will be reached.
The `WaitFor` method can use one or several conditions with the operator `&&` or `||` are used. Don't mix the both operator.

List of all conditions of the `WaitFor` method:

* **_ep**: The number of epochs before the learning stops. A epoch is one learn about the whole dataset.
* **_acc**: The minimum accuracy (between 0 and 1) to reach before the learning stops. The accuracy is equal to classification rate for classification dataset.
* **_mae**: The maximum MAE (mean absolute error) to reach before the learning stops. The ME is equal to classification rate for classification dataset.
* **_ms**: The minimum time before the learning stops in milliseconds.
* **_s**: The minimum time before the learning stops in seconds.
* **_min**: The minimum time before the learning stops in minutes.

### Tip
You can set the `verbose` enum to `minimal` for display the accuracy and the epoch number during training.
