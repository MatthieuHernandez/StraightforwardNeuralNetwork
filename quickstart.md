---
layout: default
title: Quickstart
nav_order: 3
---

## Quickstart

First, don't forget `include` and `namespace`

```cpp
#include "StraightforwardNeuralNetwork/src/neural_network/StraightforwardNeuralNetwork.hpp"
#include "StraightforwardNeuralNetwork/src/data/Data.hpp.hpp"
using namespace snn;
```

After, you just need to follow this 5 basic steps.

### 1) Prepare you data

Create a `Data` object, choose your problem type [classification]({{site.baseurl}}/data/classification.html), [multipleClassification]({{site.baseurl}}/data/multiple_classification.html) or [regression]({{site.baseurl}}/data/regression.html).
```cpp
Data data(problem::classification, inputData, expectedOutputs);
```

### 2) Design the neural network

Create a `StraightforwardNeuralNetwork` object, choose the architecture using the [layers]({{site.baseurl}}/neural_network/Layer/layer.html).
```cpp
StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, activation::ReLU),
    FullyConnected(70, activation::tanh),
    FullyConnected(10, activation::sigmoid)
});
```

### 4) Train the neural network

Train the neural network and wait until the neural network has learned.
```cpp
neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s || 0.9_acc);
neuralNetwork.stopTraining();
```

### 5) Use it

```cpp
vector<float> output = neuralNetwork.computeOutput(input); // for regression and multiple classification
```
or
```cpp
int classNumber = neuralNetwork.computeCluster(input); // for classification
```

### 6) Save and load (Optional)
```cpp
neuralNetwork.SaveAs(".\MyFirstNeuralNetwork.snn");
neuralNetwork.LoadFrom(".\MyFirstNeuralNetwork.snn");
```
