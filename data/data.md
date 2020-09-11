---
layout: default
title: Data
nav_order: 4
permalink: /data
has_children: true
has_toc: false
---
# Data &#128200;
**Data** is one of most important class of the library, this class store the data using by the neural network.

## Declaration
This is the Data constructors.
```cpp
Data(problem typeOfProblem,
     std::vector<std::vector<float>>& trainingInputs,
     std::vector<std::vector<float>>& trainingLabels,
     std::vector<std::vector<float>>& testingInputs,
     std::vector<std::vector<float>>& testingLabels,
     nature typeOfTemporal = nature::nonTemporal,
     int numberOfRecurrences = 0);
```
**Arguments**
* **typeOfProblem**: An `enum` corresponding to the type of problem associated with the data. There are 3 types of problems [classification]({{site.baseurl}}/data/classification.html), [multipleClassification]({{site.baseurl}}/data/multiple_classification.html) and [regression]({{site.baseurl}}/data/regression.html).
 * **trainingInputs**: 2D vector of all the data inputs use to train the neural network. Each `vector<float>` represents an input for the neural network. 
 * **trainingLabels**: 2D vector of all the expected outputs use to train the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **testingInputs**: 2D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network.
 * **testingLabels**: 2D vector of all the expected outputs use to evaluate the neural network. Each `vector<float>` represents the expected output by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).
 * **numberOfRecurrences**: Size of sequence used for train neural network. Only used for [timeSeries]({{site.baseurl}}/data/time_series.html) otherwise leave the value at 0.

**Each example of a  dataset must always be an 1D vector** (except for [sequential data]({{site.baseurl}}/data/sequential.html)). For example to use [CIFAR-10 dataset]({{site.baseurl}}/examples/CIFAR-10.html) each image must be converted to 1D vector. It's the neural network [Input layer] which permit to define the shape of image, here `Input(32, 32,3)`.

**Data** has a 2nd constructors if the data for training and testing are the same.
```cpp
Data(problem typeOfProblem,
     std::vector<std::vector<float>>& inputs,
     std::vector<std::vector<float>>& labels,
     nature temporal = nature::nonTemporal,
     int numberOfRecurrences = 0);
```

Here is the simplest example of declaration a data for classification problem. 
```cpp
vector<vector<float>> inputs;
vector<vector<float>> label;
Data data(problem::classification, inputData, expectedOutputs, nature::nonTemporal);
```

The **Data** allows neuron networks to solve 3 types of problem:
```cpp
enum class problem
    {
        classification,
        multipleClassification,
        regression
    };
```

Data can process data of many natures:
```cpp
    enum class nature
    {
        nonTemporal,
        sequential,
        timeSeries,
    };
```
There are 3 types of problems and 3 types of temporality.