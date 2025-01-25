#include <snn/data/Data.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"

using namespace std;
using namespace snn;

/*
This is the simplest example how to use this library
In this neural network return 3 ouputs AND, NAND, OR logical operator of 2 inputs.
For more explanation go to wiki
*/
auto multipleClassificationExample() -> int
{
    vector<vector<float>> inputData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> expectedOutputs = {{0, 1, 0}, {0, 1, 1}, {0, 1, 1}, {1, 0, 1}};

    const float separator = 0.5F;
    Data data(problem::multipleClassification, inputData, expectedOutputs);
    data.setSeparator(separator);

    StraightforwardNeuralNetwork neuralNetwork({Input(2), FullyConnected(8), FullyConnected(3)});

    neuralNetwork.train(data, 1.00_acc || 2_s);  // train neural network until 100% accuracy or 3s on a parallel thread

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    vector<float> output =
        neuralNetwork.computeOutput(data.getData(snn::testing, 0));  // consult neural network to test it

    if (accuracy == 100 && output[0] < separator && output[1] > separator && output[2] < separator &&
        neuralNetwork.isValid() == snn::ErrorType::noError)
        return EXIT_SUCCESS;  // the neural network has learned
    return EXIT_FAILURE;
}
