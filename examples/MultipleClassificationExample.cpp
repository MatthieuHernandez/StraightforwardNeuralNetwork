#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"

using namespace snn;

/*
This is the simplest example how to use this library
In this neural network return 3 ouputs AND, NAND, OR logical operator of 2 inputs.
For more explanation go to wiki
*/
auto multipleClassificationExample() -> int
{
    std::vector<std::vector<float>> inputData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<float>> expectedOutputs = {{0, 1, 0}, {0, 1, 1}, {0, 1, 1}, {1, 0, 1}};

    const float separator = 0.5F;
    Dataset dataset(problem::multipleClassification, inputData, expectedOutputs);
    dataset.setSeparator(separator);

    StraightforwardNeuralNetwork neuralNetwork({Input(2), FullyConnected(8), FullyConnected(3)});

    neuralNetwork.train(dataset, 1.00_acc || 3_s);  // Train until 100% accuracy or 3s on a parallel thread.

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    std::vector<float> output =
        neuralNetwork.computeOutput(dataset.getTestingData(0));  // consult neural network to test it

    if (accuracy == 100 && output[0] < separator && output[1] > separator && output[2] < separator &&
        neuralNetwork.isValid() == snn::errorType::noError)
    {
        return EXIT_SUCCESS;  // The neural network has learned.
    }
    return EXIT_FAILURE;
}
