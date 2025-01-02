#include <cstddef>
#include "../ExtendedGTest.hpp"
#include <snn/tools/Tools.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

using namespace std;
using namespace snn;

TEST(Train, WithoutError)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(4, activation::sigmoid),
    },
        StochasticGradientDescent(0.1f));
    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    vector<float> input = { 0.9f, 0.0f, -0.7f, 0.1f };
    auto expected = neuralNetwork.computeOutput(input);
    neuralNetwork.trainOnce(input, expected);

    auto valueAfterTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    ASSERT_EQ(valueBeforeTrain, valueAfterTrain);
}

TEST(Train, WithNanAsExpected)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(4, activation::sigmoid),
    },
        StochasticGradientDescent(0.2f));

    vector<float> input = { 1, 0, 1, 0 };
    vector<float> expected1 = { NAN, NAN, NAN, NAN };
    vector<float> expected2 = { 1, NAN, NAN, 0 };

    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected1);
    auto valueAfterTrain1 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected2);
    auto valueAfterTrain2 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    ASSERT_EQ(valueBeforeTrain, valueAfterTrain1);
    ASSERT_NE(valueBeforeTrain, valueAfterTrain2);
}