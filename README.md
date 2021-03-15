<p align="center">
    <img src="https://github.com/MatthieuHernandez/NeuralNetworkTest/blob/master/CPU_MLP.png" width="128" style="text-align:center">
    <br/>
    <h1 align="center"> Straightforward Neural Network </h1>
</p>

![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/workflows/Unit%20tests%20Linux/badge.svg?barnch=master)
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/workflows/Unit%20tests%20Windows/badge.svg?barnch=master)
![](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/workflows/Dataset%20tests/badge.svg?barnch=master)

**Straightforward Neural Network** is an open source neural network library in C++17 optimized for CPU. The goal of this library is to make the use of neural networks as easy as possible.

## Documentation
 See the full documentation [here](https://matthieuhernandez.github.io/StraightforwardNeuralNetwork/).

## Classification datasets results
| Dataset Name | Data type | Problem type | Score | Number of Parameters |
|--------------|-----------|--------------|-------|----------------------|
| [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) | audio        | classification | -                        | -      |
| [Daily min temperatures](https://github.com/jbrownlee/Datasets)            | time series  | regression     | 1.42 Mean Absolute Error | 30     |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                    | image        | classification | 52.30% Accurracy         | 196200 |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)          | image        | classification | 88.61% Accurracy         | 95874  |
| [MNIST](http://yann.lecun.com/exdb/mnist)                                  | image        | classification | 98.62% Accurracy         | 209000 |
| [Wine](https://archive.ics.uci.edu/ml/datasets/wine)                       | multivariate | classification | 100.0% Accurracy         | 444    |
| [Iris](https://archive.ics.uci.edu/ml/datasets/iris)                       | multivariate | classification | 100.0% Accurracy         | 150    |

## Installation (with *CMake* 3.17.1)

### Linux, UNIX - GCC 10.1
* To compile open a command prompt and run `cmake -G"Unix Makefiles" ./..  && make` from the `build` folder.

* To run the unit tests execute `./tests/unit_tests/UnitTests` from `build` folder.

* To run the dataset tests run `./tests/dataset_tests/ImportDatasets.sh` and execute `./tests/dataset_tests/DatasetTests` from `build` folder.

### Windows - MSVC++ 14.2
* You can generate a Visual Studio project by running `cmake -G"Visual Studio 16 2019" ./..` from `build` folder.

* To run the unit tests open `./tests/unit_tests/UnitTests.vcxproj` in Visual Studio.

* To run the dataset tests run `./tests/dataset_tests/ImportDatasets.sh` and open `./tests/dataset_tests/DatasetTests.vcxproj` in Visual Studio.

 ## Use
Create, train and use a neural network in few lines of code.
```cpp
using namespace snn;

Data data(problem::classification, inputData, expectedOutputs);

StraightforwardNeuralNetwork neuralNetwork({
    Input(28, 28, 1), 
    Convolution(1, 3, activation::ReLU),
    FullyConnected(70, activation::tanh),
    FullyConnected(10, activation::sigmoid)
});

neuralNetwork.train(data, 0.90_acc || 20_s); // train neural network on data until 90% accuracy or 30s

float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
```
[see more details](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki/)
## License

[Apache License 2.0](LICENSE)
