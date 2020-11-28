#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Recurrence, RepeatInput)
{
    vector2D<float> inputData =       {{0.0f}, {-1.0f}, {-0.8f}, {-0.5f}, {-0.2f}, {0.0f}, {0.3f}, {0.5f}, {0.7f}, {1.0f}};
    vector2D<float> expectedOutputs = {{0.0f}, {-1.0f}, {-0.8f}, {-0.5f}, {-0.2f}, {0.0f}, {0.3f}, {0.5f}, {0.7f}, {1.0f}};
    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.15f);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12),
        FullyConnected(6),
        FullyConnected(1, activation::tanh)
    },
        StochasticGradientDescent(0.003f, 0.97f));
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

TEST(Recurrence, RepeatLastInput)
{
    vector2D<float> inputData =       {{0}, {0}, {1}, {1}, {0}, {-1}, {-1}, {0},  {1}, {-1}, {1},  {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {1}, {1}, {0},  {-1}, {-1}, {0}, {1},  {-1}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    data->setPrecision(0.4f);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(20),
        FullyConnected(8),
        FullyConnected(1, activation::tanh)
    },
        StochasticGradientDescent(0.02f));

    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

//a simple recurrent neural network can't solve this problem
TEST(Recurrence, RepeatLastLastInput)
{
    vector2D<float> inputData =       {{0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {1}, {0}, {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {1}};

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 2);
    data->setPrecision(0.3f);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        GruLayer(15),
        GruLayer(5),
        FullyConnected(1)
    },
        StochasticGradientDescent(0.05f, 0.1f));

    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

inline
void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(1.0_acc || 7_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRateMax();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.5);
}
