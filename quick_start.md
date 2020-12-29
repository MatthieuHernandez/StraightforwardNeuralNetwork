---
layout: default
title: Quick start
nav_order: 3
---

# Quick start &#128640;
{: .no_toc }

First, don't forget `include` and `namespace`

```cpp
#include "StraightforwardNeuralNetwork/src/neural_network/StraightforwardNeuralNetwork.hpp"
#include "StraightforwardNeuralNetwork/src/data/Data.hpp"
using namespace snn;
```

After, you just need to follow this 5 basic steps.

1. TOC
{:toc}

### Prepare you data
Create a `Data` object, choose your problem type [classification]({{site.baseurl}}/data/classification.html), [multipleClassification]({{site.baseurl}}/data/multiple_classification.html) or [regression]({{site.baseurl}}/data/regression.html).
```cpp
Data data(problem::classification, inputData, expectedOutputs);
```

### Design the neural network
Create a `StraightforwardNeuralNetwork` object, choose the architecture using the [layers]({{site.baseurl}}/neural_network/Layer/layer.html).
```cpp
StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, activation::ReLU),
    FullyConnected(70, activation::tanh),
    FullyConnected(10, activation::sigmoid)
});
```

### Train the neural network
Train the neural network and wait until the neural network has learned.
```cpp
neuralNetwork.train(data, 20_s || 0.9_acc);
```

### Use it
Use the neural networks to predict or calculate the class of new data.
```cpp
vector<float> output = neuralNetwork.computeOutput(input); // for regression and multiple classification
```
or
```cpp
int classNumber = neuralNetwork.computeCluster(input); // for classification
```

### Save and load (Optional)
```cpp
neuralNetwork.SaveAs(".\MyFirstNeuralNetwork.snn");
neuralNetwork.LoadFrom(".\MyFirstNeuralNetwork.snn");
```

