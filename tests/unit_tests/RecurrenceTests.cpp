#include "../ExtendedGTest.hpp"
#include "tools/ExtendedExpection.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Recurence, SimplestTest)
{
    vector<vector<float>> inputData = {{-1.0}, {-0.5}, {0.0}, {0.5}, {1.0}};
    vector<vector<float>> expectedOutputs = {{-1.0}, {-0.5}, {0.0}, {0.5}, {1.0}};
    auto data = make_unique<Data>(regression, inputData, expectedOutputs, continuous, 1);
    data->setPrecision(0.1);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        AllToAll(4),
        AllToAll(1)
    });
    neuralNetwork.optimizer.learningRate = 0.005f;
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


