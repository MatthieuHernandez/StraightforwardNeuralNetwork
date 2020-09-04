---
layout: default
title: CIFAR-10
parent: Examples on datasets
nav_order: 3
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/cifar_10.png" att="CIFAR-10" width="308px" class="center"/>
</p>

# CIFAR-10

## Description

The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. There are 50,000 training images and 10,000 test images.

[See more details](https://www.cs.toronto.edu/~kriz/cifar.html)

## Neural network 

This is the test with the neural network architecture used to obtain up to **36.80%** accuracy on this dataset.
_To reach this accuracy you may need more tetative and more learning time._


```cpp
TEST_F(Cifar10Test, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3072),
        FullyConnected(200),
        FullyConnected(80),
        FullyConnected(10)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 300_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.24);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/CIFAR-10/Cifar10Test.cpp)