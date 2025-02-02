#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"

using namespace snn;

/*
This is a simple example how to use neural network for a regression.
In this neural network return the average of 3 inputs.
For more explanation go to wiki.
*/
auto regressionExample() -> int
{
    std::vector<std::vector<float>> inputData = {{0, 1, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {0, 0, 0}, {1, 1, 1}};
    std::vector<std::vector<float>> expectedOutputs = {{0.333F}, {0.666F}, {0.666F}, {0.666F}, {0.0F}, {1.0F}};

    float precision = 0.1F;
    Dataset dataset(problem::regression, inputData, expectedOutputs);
    dataset.setPrecision(precision);

    snn::StraightforwardNeuralNetwork neuralNetwork({Input(3), FullyConnected(5), FullyConnected(1)});

    neuralNetwork.train(dataset,
                        1.00_acc || 2_s);  // Train neural network until 100% accuracy or 3s on a parallel thread.

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    std::vector<float> output =
        neuralNetwork.computeOutput(dataset.getTestingData(0));  // Consult neural network to test it.
    std::vector<float> expectedOutput = dataset.getTestingOutputs(0);

    if (accuracy == 100 && std::abs(output[0] - expectedOutput[0]) < precision &&
        neuralNetwork.isValid() == errorType::noError)
    {
        return EXIT_SUCCESS;  // The neural network has learned.
    }
    return EXIT_FAILURE;
}