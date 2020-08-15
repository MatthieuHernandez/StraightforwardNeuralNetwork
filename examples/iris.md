---
layout: default
title: Iris
parent: Examples on datasets
nav_order: 7
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/iris.png" att="Iris" width="538px" class="center"/>
</p>

# Iris

## Description

The Iris flower dataset or Fisher's Iris data set is a multivariate data set introduced by Ronald Fisher in 1936.
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other

[See more details](https://archive.ics.uci.edu/ml/datasets/iris)

## Neural network 

This is the test with the neural network architecture used to obtain up to **100%** accuracy on this dataset.
:warning: _To reach this accuracy you may need more tetative and more learning time.__


```cpp
TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(15),
        FullyConnected(5),
        FullyConnected(3)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(0.98_acc || 2_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/Iris/IrisTest.cpp)