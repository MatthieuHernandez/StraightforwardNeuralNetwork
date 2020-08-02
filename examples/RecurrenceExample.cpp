#include "Examples.hpp"
#include "../tests/ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a time series.
In this neural network return the sum of 2 inputs.
For more explanation go to wiki.
*/
int recurrenceExample()
{
    vector<vector<float>> inputData = {
        {0.3}, {0.5}, {0.4}, {0.2}, {0.0}, {0.2}, {0.2}, {0.4}, {0.1}, {0.3}, {0.4}, {0.0}, {0.0}, {0.4}, {0.4}, {0.3}, {0.2}, {0.1}, {0.2}, {0.0}, {0.1}, {0.5}, {0.5}, {0.3}, {0.3}
    };
    vector<vector<float>> expectedOutputs = {
        {0.3}, {0.8}, {0.9}, {0.6}, {0.2}, {0.2}, {0.4}, {0.6}, {0.5}, {0.4}, {0.7}, {0.4}, {0.0}, {0.4}, {0.8}, {0.7}, {0.5}, {0.3}, {0.3}, {0.2}, {0.1}, {0.6}, {0.10}, {0.8}, {0.6}
    };

    const float precision = 0.5f;
    Data data(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(10, 1),
        FullyConnected(1, activation::sigmoid)
    });

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc || 3_s); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
    float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100
        && mae < precision
        && neuralNetwork.isValid() == 0)
    {
        return EXIT_SUCCESS; // the neural network has learned
    }
    return EXIT_FAILURE;
}