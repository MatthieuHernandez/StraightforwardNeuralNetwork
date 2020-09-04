---
layout: default
title: Daily min temperatures
parent: Examples on datasets
nav_order: 2
---

<p>
    <img src="{{site.baseurl}}/assets/images/examples/temperatures.jpg" att="temperatures" width="270px" class="center"/>
</p>

# Daily min temperatures

## Description


The Daily min temperatures dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia. The units are in degrees Celsius and there are 3650 observations. The source of the data is credited as the Australian Bureau of Meteorology.

[See more details](https://github.com/jbrownlee/Datasets)

## Neural network 

This is the test with the neural network architecture used to obtain up to **1.42** mean absolute error on this dataset.
_To reach this accuracy you may need more tetative and more learning time._


```cpp
TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
   StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(10),
        FullyConnected(1, activation::identity)
    });
    neuralNetwork.optimizer.learningRate = 0.004f;
    neuralNetwork.optimizer.momentum = 0.2f;
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(7_s || 2.0_mae);
    neuralNetwork.stopTraining();
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 2.0);
}
```

[See the code](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/blob/master/tests/dataset_tests/daily-min-temperatures/DailyMinTemperaturesTest.cpp)