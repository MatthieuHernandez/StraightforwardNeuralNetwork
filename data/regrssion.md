---
layout: default
title: Regression
parent: Data
nav_order: 3
---

# Regression

## Presentation
**Data** can be use to regression. Regression is used when you want specific output values. The outputs can have any values but it is better to normalize the output to 0 and 1 and use a [sigmoid]({{site.baseurl}}/neural_network/Layer/activation_functions.html) as the output enable function. (Or a [tanh]({{site.baseurl}}/neural_network/Layer/activation_functions.html) for values between -1 and 1)

## Declaration
```cpp
Data(problem:regression,
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