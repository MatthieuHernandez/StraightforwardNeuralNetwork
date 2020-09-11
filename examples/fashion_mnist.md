---
layout: default
title: Fashion-MNIST
parent: Examples on datasets
nav_order: 4
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/fashion_mnist.jpg" att="Fashion-MNIST" width="240px" class="center"/>
</p>

# Fashion-MNIST

## Description

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

[See more details](https://github.com/zalandoresearch/fashion-mnist)

## Neural network 

This is the test with the neural network architecture used to obtain up to **88.13%** accuracy on this dataset.
_To reach this accuracy you may need more attempts and more learning time._


```cpp
TEST_F(FashionMnistTest, convolutionalNeuralNetwork)
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
    ASSERT_ACCURACY(accuracy, 0.75);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/Fashion-MNIST/FashionMnistTest.cpp)