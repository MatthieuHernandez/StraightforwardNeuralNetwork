---
layout: default
title: Sequential data
parent: Data
nav_order: 5
---

# Sequential data

## Presentation
Sequential data is mostly used for classification of dataset composed of several examples of different durations. For example chess games, different sounds,videos, etc.

## Declaration
```cpp
Data(problem typeOfProblem,
     std::vector<std::vector<std::vector<float>>>& trainingInputs,
     std::vector<std::vector<float>>& trainingLabels,
     std::vector<std::vector<std::vector<float>>>& testingInputs,
     std::vector<std::vector<float>>& testingLabels,
     nature::timeSeries);
```
**Arguments**
 * **trainingInputs**: 3D vector of all the data inputs use to train the neural network. Each `vector<float>` represents an input for the neural network for one example at one moment.
 * **trainingLabels**: 2D vector of all the expected outputs use to train the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **testingInputs**: 3D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network for one example at one moment.
 * **testingLabels**: 2D vector of all the expected outputs use to evaluate the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).
 * **numberOfRecurrences**: Size of sequence used for train neural network. Only used for [timeSeries]({{site.baseurl}}/data/time_series.html) otherwise leave the value at 0.
 
The labels is a 2D vector and inputs is a 3D vector because each example is represented by a 2D vector of each vector of input at each time.