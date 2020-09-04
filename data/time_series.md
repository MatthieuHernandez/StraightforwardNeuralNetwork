---
layout: default
title: Time series
parent: Data
nav_order: 5
---

# Time series

## Presentation
Time series is used for continuous data. They are often used for the prediction of continuous temporal data such as daily temperature, the price of an product, the evolution of a population, etc. over time.

## Declaration
```cpp
Data(problem:classification,
     std::vector<std::vector<float>>& trainingInputs,
     std::vector<std::vector<float>>& trainingLabels,
     std::vector<std::vector<float>>& testingInputs,
     std::vector<std::vector<float>>& testingLabels,
     nature::timeSeries,
     int numberOfRecurrences = 0);
```
**Arguments**
 * **trainingInputs**: 2D vector of all the data inputs use to train the neural network. Each `vector<float>` represents an input for the neural network. 
 * **trainingLabels**: 2D vector of all the expected ouputs use to train the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **testingInputs**: 2D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network.
 * **testingLabels**: 2D vector of all the expected ouputs use to evaluate the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).
 * **numberOfRecurrences**: Size of sequence used for train neural network. Only used for [timeSeries]({{site.baseurl}}/data/time_series.html) otherwise leave the value at 0.
 
 Here the number of recurrences is very import because it determines the amount of data to be used by the neural network to predict during the learning phase. If the value is too low the neuron networks will not have enough information to learn and if the value is too high it will unnecessarily lengthen the learning time.