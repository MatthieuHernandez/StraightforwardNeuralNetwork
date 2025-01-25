#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Recurrence, RepeatInput)
{
    vector2D<float> inputData = {{0.0F}, {-1.0F}, {-0.8F}, {-0.5F}, {-0.2F}, {0.0F}, {0.3F}, {0.5F}, {0.7F}, {1.0F}};
    vector2D<float> expectedOutputs = {{0.0F}, {-1.0F}, {-0.8F}, {-0.5F}, {-0.2F},
                                       {0.0F}, {0.3F},  {0.5F},  {0.7F},  {1.0F}};
    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.15F);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), Recurrence(12), FullyConnected(6), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.003F, 0.97F));
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

TEST(Recurrence, RepeatLastInput)
{
    vector2D<float> inputData = {{0}, {0}, {1}, {1}, {0}, {-1}, {-1}, {0}, {1}, {-1}, {1}, {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {1}, {1}, {0}, {-1}, {-1}, {0}, {1}, {-1}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.4F);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), Recurrence(20), FullyConnected(8), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.02F, 0.5F));
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

// a simple recurrent neural network can't solve this problem
TEST(Recurrence, RepeatLastLastInput)
{
    vector2D<float> inputData = {{0.2F}, {0}, {1},    {0}, {1}, {1}, {0.1F}, {0}, {1}, {1},
                                 {0.9F}, {1}, {0.3F}, {0}, {1}, {0}, {1},    {0}, {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0.2F}, {0}, {1},    {0}, {1}, {1}, {0}, {0.1F},
                                       {1}, {1}, {0.9F}, {1}, {0.3F}, {0}, {1}, {0}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 2);
    data->setPrecision(0.3F);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), GruLayer(14), GruLayer(10), FullyConnected(1)},
                                               StochasticGradientDescent(0.1F, 0.9F));

    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

inline void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.train(d, 1.0_acc || 4_s, 1, 10);
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.5);
}
