---
layout: default
title: Training
parent: Neural network
nav_order: 4
---

# Training &#127947;
This part is about how to train a `StraightforwardNeuralNetwork`. There are 2 ways to do this, synchronous with the `train` method or async with the methods `startTrainingAsync` and `stopTrainingAsync`.

## Declaration 
```cpp
void train(Data& data, Wait wait, int batchSize = 1, int evaluationFrequency = 1);
```
**Arguments**
 * **data**: The dataset used to train the neural network
 * **wait**: Stop training condition
 * **batchsize**: The size of batch
* **evaluationFrequency**: The number of epochs between each evaluation of the neural network on the testing set

## Example 
### The synchronous version
```cpp
StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(20), FullyConnected(3)});
neuralNetwork.train(data, 20_s || 0.9_acc); // train neural network on data until 90% accuracy or 20s
```
### The asynchronous version
```cpp
StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(20), FullyConnected(3)});
neuralNetwork.startTrainingAsync(data); // start training on a new parallel thread
neuralNetwork.waitFor(20_s || 0.9_acc); // wait until the network reach 90% accuracy or 20s
neuralNetwork.stopTrainingAsync(); // stop training, evaluate the network and delete the thread
```
It is better to use the `train` method unless you do something else during the training such as displaying a real-time graph.

List of all conditions for stop the `train` or `WaitFor` methods:

* **_ep**: The number of epochs to reach. A epoch is one learn about the whole dataset.
* **_acc**: The minimum accuracy (between 0 and 1) to reach.
* **_mae**: The maximum MAE (mean absolute error) to reach.
* **_ms**: Time before the learning stops in milliseconds.
* **_s**: Time before the learning stops in seconds.
* **_min**: Time before the learning stops in minutes.

You can use one or several conditions with the operator `&&` or `||` are used but you can't mix the both operator.

## Batch size and evaluation frequency 
You can add bach size and change the evaluation frequency. The default value for batch size and evaluation frequency is 1.
You can set the evaluation frequency to 0 to never evaluate the network during learning, therefore the stop condition must be a duration.
```cpp
neuralNetwork.train(data, 1.5_mae, 32, 2); // set the bach size to 32 and the evaluation frequency to 2
```


### Tip
You can set the `verbose` enum to `minimal` for display the accuracy and the epoch number during training.
