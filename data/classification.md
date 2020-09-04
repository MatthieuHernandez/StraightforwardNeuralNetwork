---
layout: default
title: Classification
parent: Data
nav_order: 1
---

# Classification

## Presentation

**Data** can be use to classification problem. For classification the label vector for an input vector must be a vector of 0 with just a 1 for the class number. For example if you have 5 classes and the input corresponds to class 3 the expected ouput vector must be:
```cpp
vector<float> expectedOutput = {0, 0, 1, 0, 0};
```
 When calculating accuracy of a neural network the class of items is the output with the value closest to 1. 

## Declaration
```cpp
Data(problem:classification,
     std::vector<std::vector<float>>& trainingInputs,
     std::vector<std::vector<float>>& trainingLabels,
     std::vector<std::vector<float>>& testingInputs,
     std::vector<std::vector<float>>& testingLabels,
     nature typeOfTemporal = nature::nonTemporal,
     int numberOfRecurrences = 0);
```
**Arguments**
 * **trainingInputs**: 2D vector of all the data inputs use to train the neural network. Each `vector<float>` represents an input for the neural network. 
 * **trainingLabels**: 2D vector of all the expected ouputs use to train the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **testingInputs**: 2D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network.
 * **testingLabels**: 2D vector of all the expected ouputs use to evaluate the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).
 * **numberOfRecurrences**: Size of sequence used for train neural network. Only used for [timeSeries]({{site.baseurl}}/data/time_series.html) otherwise leave the value at 0.

