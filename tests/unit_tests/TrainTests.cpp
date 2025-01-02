#include <cstddef>
#include "../ExtendedGTest.hpp"
#include <snn/tools/Tools.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

using namespace std;
using namespace snn;

TEST(Train, WithNanAsExpected)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(4, activation::sigmoid),
        },
        StochasticGradientDescent(0.1f));

    vector<float> input = { 1, 0, 1, 0 };
    vector<float> expected1 = { NAN, NAN, NAN, NAN };
    vector<float> expected2 = { 1, NAN, NAN, 0 };

    auto valueBeforeTrain = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected1);
    auto valueAfterTrain1 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();
    neuralNetwork.trainOnce(input, expected2);
    auto valueAfterTrain2 = neuralNetwork.layers[0]->getAverageOfAbsNeuronWeights();

    ASSERT_EQ(valueBeforeTrain, valueAfterTrain1);
    ASSERT_EQ(valueBeforeTrain, valueAfterTrain2);
}