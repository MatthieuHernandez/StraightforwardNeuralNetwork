<p align="center">
    <img src="https://github.com/MatthieuHernandez/NeuralNetworkTest/blob/master/CPU_MLP.png" width="128" style="text-align:center">
    <br/>
    <h1 align="center"> Straightforward Neural Network </h1>
</p>

[![All tests - GCC Linux](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/gcc_linux.yml/badge.svg)](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/gcc_linux.yml)
[![Unit tests - Clang Linux](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/clang_linux.yml/badge.svg)](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/clang_linux.yml)
[![Unit tests - MSVC Windows](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/msvc_windows.yml/badge.svg)](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/msvc_windows.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/matthieuhernandez/straightforwardneuralnetwork/badge)](https://www.codefactor.io/repository/github/matthieuhernandez/straightforwardneuralnetwork)

**Straightforward Neural Network** is an open source neural network library in C++20 optimized for CPU. The goal of this library is to make the use of neural networks as easy as possible.

## Documentation
 See the full documentation [here](https://matthieuhernandez.github.io/StraightforwardNeuralNetwork/).

## Classification datasets results
| Dataset Name | Data type | Problem type | Score | Number of Parameters |
|--------------|-----------|--------------|-------|----------------------|
| [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) | audio        | classification | 91.05% Accurracy         | 3082   |
| [Daily min temperatures](https://github.com/jbrownlee/Datasets)            | time series  | regression     | 1.42 Mean Absolute Error | 30     |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                    | image        | classification | 66.72% Accurracy         | 365548 |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)          | image        | classification | 89.55% Accurracy         | 270926 |
| [MNIST](http://yann.lecun.com/exdb/mnist)                                  | image        | classification | 99.00% Accurracy         | 83246  |
| [Wine](https://archive.ics.uci.edu/ml/datasets/wine)                       | multivariate | classification | 100.0% Accurracy         | 444    |
| [Iris](https://archive.ics.uci.edu/ml/datasets/iris)                       | multivariate | classification | 100.0% Accurracy         | 150    |

## Installation (with *CMake* 3.17.1)

* First of all, move to a build folder: `mkdir build && cd ./build`
* For dataset tests, the datasets must be downloaded: `./resources/ImportDatasets.sh`

### Linux, UNIX - GCC 14.2.0 or Clang 18.1.3

* Use CMake to build: `cmake -G"Unix Makefiles" ./..  && make`

* Run the unit tests: `./build/tests/unit_tests/UnitTests`

* Run the dataset tests: `./tests/dataset_tests/DatasetTests`

### Windows - MSVC 19.41

* Use CMake to generate a VS2022 project : `cmake -G"Visual Studio 17 2022" ./..`

* To run the unit tests open the generated project: `./build/tests/unit_tests/UnitTests.vcxproj`

* To run the dataset tests open the generated project: `./build/tests/dataset_tests/DatasetTests.vcxproj`

 ## Use
Create, train and use a neural network in few lines of code.
```cpp
#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

snn::Dataset dataset(snn::problem::classification, inputData, expectedOutputs);

snn::StraightforwardNeuralNetwork neuralNetwork({
    Input(1, 28, 28),  // The input shape is (C, X, Y).
    Convolution(16, 3, activation::ReLU),  // The layer has 16 filters and (3, 3) kernels.
    FullyConnected(92),  // The layer has 92 neurons.
    FullyConnected(10, activation::identity, Softmax())
});

neuralNetwork.train(dataset, 0.90_acc || 20_s);  // Train neural network on data until 90% accuracy or 20s.

float accuracy = neuralNetwork.getGlobalClusteringRate();  // Retrieve the accuracy.
```
[see more details](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki/)
## License

[Apache License 2.0](LICENSE)
