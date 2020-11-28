#include "Examples.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a regression.
In this neural network return the average of 3 inputs.
For more explanation go to wiki.
*/
int regressionExample()
{
    vector<vector<float>> inputData = {{0, 1, 0}, {0, 1, 1},{1, 0, 1}, {1, 1, 0}, {0, 0, 0}, {1, 1, 1}};
    vector<vector<float>> expectedOutputs = {{0.333f}, {0.666f}, {0.666f}, {0.666f}, {0.0f}, {1.0f}};

    float precision = 0.1f;
    Data data(problem::regression, inputData, expectedOutputs);
    data.setPrecision(precision);

    snn::StraightforwardNeuralNetwork neuralNetwork({Input(3), FullyConnected(5), FullyConnected(1)});

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc || 3_s ); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0f;
    vector<float> output = neuralNetwork.computeOutput(data.getData(snn::testing, 0)); // consult neural network to test it
    vector<float> expectedOutput = data.getOutputs(snn::testing, 0);

    if (accuracy == 100
        && abs(output[0] - expectedOutput[0]) < precision
        && neuralNetwork.isValid() == 0)
    {
        return EXIT_SUCCESS; // the neural network has learned
    }
    return EXIT_FAILURE;
}