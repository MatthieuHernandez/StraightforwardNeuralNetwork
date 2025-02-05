#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"

using namespace snn;

/*
This is a simple example how to use neural network for a time series.
In this neural network return the sum of 2 inputs.
For more explanation go to wiki.
*/
auto recurrenceExample() -> int
{
    std::vector<std::vector<float>> inputData = {{0.3F}, {0.5F}, {0.4F}, {0.2F}, {0.0F}, {0.2F}, {0.2F}, {0.4F}, {0.1F},
                                                 {0.3F}, {0.4F}, {0.0F}, {0.0F}, {0.4F}, {0.4F}, {0.3F}, {0.2F}, {0.1F},
                                                 {0.2F}, {0.0F}, {0.1F}, {0.5F}, {0.5F}, {0.3F}, {0.3F}};
    std::vector<std::vector<float>> expectedOutputs = {
        {0.3F}, {0.8F}, {0.9F}, {0.6F}, {0.2F}, {0.2F}, {0.4F}, {0.6F}, {0.5F}, {0.4F},  {0.7F}, {0.4F}, {0.0F},
        {0.4F}, {0.8F}, {0.7F}, {0.5F}, {0.3F}, {0.3F}, {0.2F}, {0.1F}, {0.6F}, {0.10F}, {0.8F}, {0.6F}};

    const float precision = 0.5F;
    Dataset dataset(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    dataset.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(10), FullyConnected(1, activation::sigmoid)},
                                               StochasticGradientDescent(0.01F, 0.8F));

    neuralNetwork.train(dataset, 1.00_acc || 3_s);  // Train until 100% accuracy or 3s on a parallel thread.

    const float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    const float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100 && mae < precision && neuralNetwork.isValid() == errorType::noError)
    {
        return EXIT_SUCCESS;  // The neural network has learned.
    }
    return EXIT_FAILURE;
}
