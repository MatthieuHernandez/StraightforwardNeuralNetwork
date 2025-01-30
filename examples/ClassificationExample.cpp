#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Examples.hpp"

using namespace snn;

/*
This is a simple example how to use neural network for a classification problem.
The neural network return class 0 if sum of inputs is negative and class 1 il sum of input is positive.
For more explanation go to wiki.
*/
auto classificationExample() -> int
{
    std::vector<std::vector<float>> inputData = {{-0.1F, 0.4F, -0.6F}, {0.5F, -0.4F, -0.8F}, {-0.7F, 0.9F, -0.7F},
                                                 {-0.9F, -0.5F, 1.7F}, {0.5F, -0.5F, 0.9F},  {0.3F, 0.6F, 0.8F}};
    std::vector<std::vector<float>> expectedOutputs = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}};

    Dataset dataset(snn::problem::classification, inputData, expectedOutputs);

    snn::StraightforwardNeuralNetwork neuralNetwork({snn::Input(3), snn::FullyConnected(5), snn::FullyConnected(2)});

    neuralNetwork.train(dataset,
                        1.00_acc || 2_s);  // Train neural network until 100% accuracy or 3s on a parallel thread.

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    int classNumber = neuralNetwork.computeCluster(dataset.getTestingData(0));  // consult neural network to test it
    int expectedClassNumber = dataset.getTestingLabel(0);  // return position of neuron with highest output
    if (accuracy == 100 && classNumber == expectedClassNumber && neuralNetwork.isValid() == snn::errorType::noError)
    {
        return EXIT_SUCCESS;  // The neural network has learned.
    }
    return EXIT_FAILURE;
}