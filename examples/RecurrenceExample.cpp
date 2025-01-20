#include <snn/data/Data.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"


using namespace std;
using namespace snn;

/*
This is a simple example how to use neural network for a time series.
In this neural network return the sum of 2 inputs.
For more explanation go to wiki.
*/
int recurrenceExample()
{
    vector<vector<float>> inputData = {{0.3f}, {0.5f}, {0.4f}, {0.2f}, {0.0f}, {0.2f}, {0.2f}, {0.4f}, {0.1f},
                                       {0.3f}, {0.4f}, {0.0f}, {0.0f}, {0.4f}, {0.4f}, {0.3f}, {0.2f}, {0.1f},
                                       {0.2f}, {0.0f}, {0.1f}, {0.5f}, {0.5f}, {0.3f}, {0.3f}};
    vector<vector<float>> expectedOutputs = {{0.3f}, {0.8f}, {0.9f}, {0.6f}, {0.2f},  {0.2f}, {0.4f}, {0.6f}, {0.5f},
                                             {0.4f}, {0.7f}, {0.4f}, {0.0f}, {0.4f},  {0.8f}, {0.7f}, {0.5f}, {0.3f},
                                             {0.3f}, {0.2f}, {0.1f}, {0.6f}, {0.10f}, {0.8f}, {0.6f}};

    const float precision = 0.5f;
    Data data(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(10), FullyConnected(1, activation::sigmoid)},
                                               StochasticGradientDescent(0.01f, 0.8f));

    neuralNetwork.train(data, 1.00_acc || 2_s);  // train neural network until 100% accuracy or 3s on a parallel thread

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0f;
    float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100 && mae < precision && neuralNetwork.isValid() == ErrorType::noError)
    {
        return EXIT_SUCCESS;  // the neural network has learned
    }
    return EXIT_FAILURE;
}
