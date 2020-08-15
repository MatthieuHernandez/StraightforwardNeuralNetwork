---
layout: default
title: Quickstart
nav_order: 3
---

## After installation what to do?

First, don't forget `include` and `namespace`

```cpp
#include "StraightforwardNeuralNetwork/src/neural_network/StraightforwardNeuralNetwork.hpp"
#include "StraightforwardNeuralNetwork/src/neural_network/StraightforwardNeuralNetwork.hpp"

using namespace snn;
```

After, you just need to follow this 5 basic steps.

### 1) Prepare you data

Create a `Data` object,  choose your problem type regression, classification, multiple classification 
```cpp
Data data(problem::classification, inputData, expectedOutputs);
```

### 2) Design the neural network

```cpp
StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, activation::ReLU),
    FullyConnected(70, activation::tanh),
    FullyConnected(10, activation::sigmoid)
});
```

### 4) Train the neural network

```cpp
neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s);
neuralNetwork.stopTraining();
```

### 5) Use it

```cpp
neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s);
neuralNetwork.stopTraining();
```

### 6) Save and load (Optional)
```cpp
neuralNetwork.SaveAs(".\MyFirstNeuralNetwork.snn");
neuralNetwork.LoadFrom(".\MyFirstNeuralNetwork.snn");
```

