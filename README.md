# StraightforwardNeuralNetwork
**Straightforward Neural Network** is an open source software simple neural network library in C++

## Documentation

You can see documentation on [Wiki](https://github.com/MatthieuHernandez/StraightforwardNeuralNetwork/wiki).

## Classification datasets results
| Dataset Name | Link                                         | Accuracy | Number of Neurones |
|--------------|----------------------------------------------|----------|--------------------|
| CIFAR-10     | https://www.cs.toronto.edu/~kriz/cifar.html  | 36.8%    | 230                |
| Mnist        | http://yann.lecun.com/exdb/mnist/            | 98.32%   | 230                |
| Wine         | https://archive.ics.uci.edu/ml/datasets/wine | 100%     | 28                 |
| Iris         | https://archive.ics.uci.edu/ml/datasets/iris | 100%     | 12                 |

## Installation

* To compile the library run `cmake -G"Unix Makefiles" ./..  && make` from `StraightforwardNeuralNetwork/build`.

* To run unit tests exectute `StraightforwardNeuralNetwork/build/tests/units_tests/UnitTests.out`

* To run dataset tests run `./ImportDatasets.sh` from `StraightforwardNeuralNetwork\tests\dataset_tests` and exectute `StraightforwardNeuralNetwork/build/tests/dataset_tests/DatasetTests.out`

## License

[Apache License 2.0](LICENSE)
