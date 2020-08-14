---
layout: default
title: Quickstart
nav_order: 2
---

## After installation what to do?


You need to follow 5 step

### 1) Prepare you data

Create a `Data` object,  choose your problem type regression, classification, multiple classification 
```
Data data(problem::classification, inputData, expectedOutputs);
```

### 2) Design the neural network

```
StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, activation::ReLU),
    FullyConnected(70, activation::tanh),
    FullyConnected(10, activation::sigmoid)
});
```

### 3) Train the neural network

```
neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s);
neuralNetwork.stopTraining();
```

