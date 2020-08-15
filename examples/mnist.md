---
layout: default
title: MNIST
parent: Examples on datasets
nav_order: 5
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/mnist.png" att="MNIST" width="240px" class="center"/>
</p>

# MNIST

## Description

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

[See more details](http://yann.lecun.com/exdb/mnist)

## Neural network 

This is the test with the neural network architecture used to obtain up to **98.32%** accuracy on this dataset.
:warning: _To reach this accuracy you may need more tetative and more learning time.__


```cpp
TEST_F(MnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(1, 5),
        FullyConnected(70),
        FullyConnected(10)
        });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.93f);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/MNIST/MnistTest.cpp)