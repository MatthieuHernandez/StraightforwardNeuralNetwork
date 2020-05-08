# StraightforwardNeuralNetwork 
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/workflows/Unit%20tests%20Linux/Windows/badge.svg?barnch=master) ![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/workflows/Dataset%20tests%20Linux/badge.svg?barnch=master)

**Straightforward Neural Network** is an open source software simple neural network library in C++

## Documentation

You can see documentation on [Wiki](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki).

## Classification datasets results
| Dataset Name  | Link                                             | Accuracy | Number of Neurones |
|---------------|--------------------------------------------------|----------|--------------------|
| CIFAR-10      | https://www.cs.toronto.edu/~kriz/cifar.html      | 36.8%    | 230                |
| Fashion-MNIST | https://github.com/zalandoresearch/fashion-mnist | 88.13%   | 230                |
| MNIST         | http://yann.lecun.com/exdb/mnist/                | 98.32%   | 230                |
| Wine          | https://archive.ics.uci.edu/ml/datasets/wine     | 100%     | 28                 |
| Iris          | https://archive.ics.uci.edu/ml/datasets/iris     | 100%     | 12                 |

## Installation (with *CMake* 3.17.1)

### Linux, UNIX - GCC 9.2

* To compile the library run `cmake -G"Unix Makefiles" ./..  && make` from `StraightforwardNeuralNetwork/build`

* To run unit tests exectute `./tests/unit_tests/UnitTests.out` from `StraightforwardNeuralNetwork/build`

* To run dataset tests run `./ImportDatasets.sh` from `StraightforwardNeuralNetwork\tests\dataset_tests` and exectute `./tests/dataset_tests/DatasetTests.out` from `StraightforwardNeuralNetwork/build`
### Windows - MSVC++ 14.2
 * You can generate a Visual Studio project: `cmake -G"Visual Studio 16 2019" ./..`
 
 ## Use
Create, train and use a neural network in few lines of code.
```cpp
Data data(classification, inputData, expectedOutputs);
StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, ReLU),
    AllToAll(70, tanh),
    AllToAll(10, sigmoid)
});

neuralNetwork.startTraining(data);
neuralNetwork.waitFor(20_s);
neuralNetwork.stopTraining();

float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
```
[see more details](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki/)
## License

[Apache License 2.0](LICENSE)
