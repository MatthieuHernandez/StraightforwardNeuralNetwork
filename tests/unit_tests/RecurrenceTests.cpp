#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Recurence, RepeatInput)
{
    vector2D<float> inputData =       {{0.0}, {-1.0}, {-0.8}, {-0.5}, {-0.2}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}};
    vector2D<float> expectedOutputs = {{0.0}, {-1.0}, {-0.8}, {-0.5}, {-0.2}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}};
    auto data = make_unique<Data>(regression, inputData, expectedOutputs, timeSeries, 1);
    data->setPrecision(0.15);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12,  1,sigmoid),
        FullyConnected(6),
        FullyConnected(1, snn::tanh)
    });
    neuralNetwork.optimizer.learningRate = 0.03f;
    neuralNetwork.optimizer.momentum = 0.98f;
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

TEST(Recurence, RepeatLastInput)
{
    vector2D<float> inputData =       {{0}, {0}, {1}, {1}, {0}, {-1}, {-1}, {0},  {1}, {-1}, {1},  {0}};
    vector2D<float> expectedOutputs = {{0}, {0}, {0}, {1}, {1}, {0},  {-1}, {-1}, {0}, {1},  {-1}, {1}};

    auto data = make_unique<Data>(regression, inputData, expectedOutputs, timeSeries, 1);
    data->setPrecision(0.3);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12, 1,snn::tanh),
        FullyConnected(6),
        FullyConnected(1, snn::tanh)
    });
    neuralNetwork.optimizer.learningRate = 0.01f;
    neuralNetwork.optimizer.momentum = 0.4f;
    testNeuralNetworkForRecurrence(neuralNetwork, *data);
}

inline
void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(1.0_acc || 5_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.5);
}
