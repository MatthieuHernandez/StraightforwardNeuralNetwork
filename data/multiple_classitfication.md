---
layout: default
title: Multiple classification
parent: Data
nav_order: 2
---

# Multiple classification

## Presentation
Multiple classification is very similar to the simple classification. The only difference is that an single data can have several classes. For example if you have 5 classes and the input corresponds to class  1 and 3 the expected output vector must be:
```cpp
vector<float> expectedOutput = {1, 0, 1, 0, 0};
```
 Unlike simple classification when calculating accuracy of a neural network all classes must be correct for the item to be considered well classified.


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
 * **trainingLabels**: 2D vector of all the expected outputs use to train the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **testingInputs**: 2D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network.
 * **testingLabels**: 2D vector of all the expected outputs use to evaluate the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).
 * **numberOfRecurrences**: Size of sequence used for train neural network. Only used for [timeSeries]({{site.baseurl}}/data/time_series.html) otherwise leave the value at 0.