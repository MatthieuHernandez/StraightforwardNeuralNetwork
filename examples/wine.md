---
layout: default
title: Wine
parent: Examples on datasets
nav_order: 6
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/wine.jpg" att="Wine" width="200px" class="center"/>
</p>

# Wine

## Description

The Wine dataset are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

[See more details](https://archive.ics.uci.edu/ml/datasets/wine)

## Neural network 

This is the test with the neural network architecture used to obtain up to **100%** accuracy on this dataset.
_To reach this accuracy you may need more tetative and more learning time._


```cpp
TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(13),
        FullyConnected(20),
        FullyConnected(8),
        FullyConnected(3)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1.00_acc || 3_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/Wine/WineTest.cpp)