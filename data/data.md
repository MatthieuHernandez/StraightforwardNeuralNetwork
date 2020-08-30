---
layout: default
title: Data
nav_order: 4
permalink: /data
has_children: true
---
# Data
**Data** is one of most important class of the library, this class store the data using by the neural network.

## Declaration
This is the Data constructors.
```cpp
Data(problem typeOfProblem,
     std::vector<std::vector<float>>& inputs,
     std::vector<std::vector<float>>& labels,
     nature temporal = nature::nonTemporal,
     int numberOfRecurrences = 0);
```
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
 * **trainingLabels**: 2D vector of all the expected ouputs use to train the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **testingInputs**: 2D vector of all the data inputs use to evaluate the neural network. Each `vector<float>` represents an input for the neural network.
 * **testingLabels**: 2D vector of all the expected ouputs use to evaluate the neural network. Each `vector<float>` represents the expected ouput by the neural network for the corresponding input.
 * **typeOfTemporal**: An `enum` corresponding to the temporal nature of problem associated with the data. There are 3 types of temporal nature [nonTemporal]({{site.baseurl}}/data/non_temporal.html), [sequential]({{site.baseurl}}/data/sequential.html) and [timeSeries]({{site.baseurl}}/data/time_series.html).

**Data** has 2 constructors, one if you have same data for training and testing and one if training data and testing data are different like on [MNIST](http://yann.lecun.com/exdb/mnist) dataset.

Here is the simplest example of declaration a data for classification problem. 
```cpp
vector<vector<float>> inputs;
vector<vector<float>> label;
Data data(problem::classification, inputData, expectedOutputs, nature::nonTemporal);
```
For classification the label vector for an input vector must be a vector of 0 with just a 1 for the class number.
For example if you have 5 classes and the input corresponds to class 3 the expected ouput vector must be:
```cpp
vector<float> expectedOutput = {0, 0, 1, 0, 0};
```

```cpp
enum class problem
    {
        classification,
        multipleClassification,
        regression
    };
```
```cpp
    enum class nature
    {
        nonTemporal,
        sequential,
        timeSeries,
    };
```
There are 3 types of problems and 3 types of temporality.


{:toc}

