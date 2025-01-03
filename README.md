<p align="center">
    <img src="https://github.com/MatthieuHernandez/NeuralNetworkTest/blob/master/CPU_MLP.png" width="128" style="text-align:center">
    <br/>
    <h1 align="center"> Straightforward Neural Network </h1>
</p>

![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/unit_tests_gcc_linux.yml/badge.svg?barnch=master)
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/unit_tests_clang_linux.yml/badge.svg?barnch=master)
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/unit_tests_windows.yml/badge.svg?barnch=master)
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/actions/workflows/dataset_tests.yml/badge.svg?barnch=master)
![](https://www.codefactor.io/repository/github/matthieuhernandez/straightforwardneuralnetwork/badge/master)

**Straightforward Neural Network** is an open source neural network library in C++20 optimized for CPU. The goal of this library is to make the use of neural networks as easy as possible.

## Documentation
 See the full documentation [here](https://matthieuhernandez.github.io/StraightforwardNeuralNetwork/).

## Classification datasets results
| Dataset Name | Data type | Problem type | Score | Number of Parameters |
|--------------|-----------|--------------|-------|----------------------|
| [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) | audio        | classification | 91.04% Accurracy         | 382    |
| [Daily min temperatures](https://github.com/jbrownlee/Datasets)            | time series  | regression     | 1.42 Mean Absolute Error | 30     |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                    | image        | classification | 61.96% Accurracy         | 207210 |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)          | image        | classification | 89.65% Accurracy         | 270926 |
| [MNIST](http://yann.lecun.com/exdb/mnist)                                  | image        | classification | 98.71% Accurracy         | 261206 |
| [Wine](https://archive.ics.uci.edu/ml/datasets/wine)                       | multivariate | classification | 100.0% Accurracy         | 444    |
| [Iris](https://archive.ics.uci.edu/ml/datasets/iris)                       | multivariate | classification | 100.0% Accurracy         | 150    |

## Installation (with *CMake* 3.17.1)

### Linux, UNIX - GCC 13.1.0 or Clang 18.1.3
* To compile open a command prompt and run `cmake -G"Unix Makefiles" ./..  && make` from the `build` folder.

* To run the unit tests execute `./tests/unit_tests/UnitTests` from `build` folder.

* To run the dataset tests run `./ImportDatasets.sh` and execute `./tests/dataset_tests/DatasetTests` from `build` folder.

### Windows - MSVC 19.41
* You can generate a Visual Studio project by running `cmake -G"Visual Studio 17 2022" ./..` from `build` folder.

* To run the unit tests open `./build/tests/unit_tests/UnitTests.vcxproj` in Visual Studio.

* To run the dataset tests run `./build/ImportDatasets.sh` and open `./build/tests/dataset_tests/DatasetTests.vcxproj` in Visual Studio.

 ## Use
Create, train and use a neural network in few lines of code.
```cpp
using namespace snn;

Data data(problem::classification, inputData, expectedOutputs);

StraightforwardNeuralNetwork neuralNetwork({
    Input(1, 28, 28), // (C, X, Y)
    Convolution(16, 3, activation::ReLU), // 16 filters and (3, 3) kernels
    FullyConnected(92),
    FullyConnected(10, activation::identity, Softmax())
});

neuralNetwork.train(data, 0.90_acc || 20_s); // Train neural network on data until 90% accuracy or 20s

float accuracy = neuralNetwork.getGlobalClusteringRate(); // Retrieve the accuracy
```
[see more details](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki/)
## License

[Apache License 2.0](LICENSE)
