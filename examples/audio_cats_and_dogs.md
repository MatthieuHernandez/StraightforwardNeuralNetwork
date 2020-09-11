---
layout: default
title: Audio Cats and Dogs
parent: Examples on datasets
nav_order: 1
---

<p >
    <img src="{{site.baseurl}}/assets/images/examples/cat_and_dog.jpg" att="cat and dog" width="280px" class="center"/>
</p>

# Audio Cats and Dogs

## Description

The Audio Cats and Dogs dataset contains 277 barking and meowing audio files from cats and dogs.

[See more details](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs)

## Neural network 

This is the test with the neural network architecture used to obtain up to **--%** accuracy on this dataset.
:warning: _To reach this accuracy you may need more attempts and more learning time._


```cpp
TEST_F(AudioCatsAndDogsTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(sizeOfOneData),
        LocallyConnected(1, 1000, activation::tanh),
        GruLayer(20),
        GruLayer(5),
        FullyConnected(2)
    });
    neuralNetwork.optimizer.learningRate = 0.002f;
    neuralNetwork.optimizer.momentum = 0.2f;
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(100_ep || 0.6_acc || 30_s);
    neuralNetwork.stopTraining();
    auto recall = neuralNetwork.getWeightedClusteringRate();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_RECALL(recall, 0.50);
    ASSERT_ACCURACY(accuracy, 0.6);
}
```
[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/audio-cats-and-dogs/AudioCatsAndDogsTest.cpp)
<br/>
_This dataset is not executed during the iteration tests because no stable architecture enough accurate was found._