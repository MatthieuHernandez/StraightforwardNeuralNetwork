#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

TEST(Train, WithoutError)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(4),
            FullyConnected(4, activation::sigmoid),
        },
        StochasticGradientDescent(0.1F));
    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    const std::vector<float> input = {0.9F, 0.0F, -0.7F, 0.1F};
    auto expected = neuralNetwork.computeOutput(input);
    neuralNetwork.trainOnce(input, expected);

    auto valueAfterTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    ASSERT_EQ(valueBeforeTrain, valueAfterTrain);
}

TEST(Train, WithNanAsExpected)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(4),
            FullyConnected(4, activation::sigmoid),
        },
        StochasticGradientDescent(0.2F));

    const std::vector<float> input = {1, 0, 1, 0};
    const std::vector<float> expected1 = {NAN, NAN, NAN, NAN};
    const std::vector<float> expected2 = {1, NAN, NAN, 0};

    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected1);
    auto valueAfterTrain1 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected2);
    auto valueAfterTrain2 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    ASSERT_EQ(valueBeforeTrain, valueAfterTrain1);
    ASSERT_NE(valueBeforeTrain, valueAfterTrain2);
    ASSERT_FALSE(std::isnan(valueAfterTrain2));
}

TEST(Train, WithWeightingAsExpected)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(4),
            FullyConnected(4, activation::sigmoid),
        },
        StochasticGradientDescent(0.2F));

    const std::vector<float> input = {1, 0, 1, 0};
    const std::vector<float> expected1 = {0, 1, 0, 0};
    const std::vector<float> expected2 = {1, 0, 0, 0};
    const std::vector<float> weighting1 = {0, 0, 0, 0};
    const std::vector<float> weighting2 = {0.1, 0.5, 0.7, 1.9};

    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected1, weighting1);
    auto valueAfterTrain1 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected2, weighting2);
    auto valueAfterTrain2 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    ASSERT_EQ(valueBeforeTrain, valueAfterTrain1);
    ASSERT_NE(valueBeforeTrain, valueAfterTrain2);
    ASSERT_FALSE(std::isnan(valueAfterTrain2));
}
