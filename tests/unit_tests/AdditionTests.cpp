#include <cstddef>
#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForAdditionTests();
unique_ptr<Data> createRecurrentDataForAdditionTests(int numberOfData);
void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Addition, WithMPL)
{
    unique_ptr<Data> data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        FullyConnected(6, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithCNN)
{
    auto data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        Convolution(6, 1, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithLCNN)
{
    auto data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        LocallyConnected(6, 1, sigmoid),
        FullyConnected(1, snn::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithRNN)
{
    auto data = createRecurrentDataForAdditionTests(400);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12, 1, sigmoid),
        FullyConnected(6, sigmoid),
        FullyConnected(1, sigmoid)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(1.0_acc || 4_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, d.getPrecision());
}

unique_ptr<Data> createDataForAdditionTests()
{
    vector2D<float> inputData = {
        {3, 5}, {5, 4}, {4, 2}, {2, 0}, {0, 2},
        {2, 4}, {4, 1}, {1, 4}, {4, 3}, {3, 0},
        {0, 0}, {0, 4}, {4, 3}, {3, 2}, {2, 1},
        {1, 2}, {2, 0}, {0, 1}, {1, 2}, {5, 5},
        {5, 3}, {1, 1}, {4, 4}, {3, 3}, {2, 2}
    };
    vector2D<float> expectedOutputs = {
        {8}, {9}, {6}, {2}, {2},
        {6}, {5}, {5}, {7}, {3},
        {0}, {4}, {7}, {5}, {3},
        {3}, {2}, {1}, {3}, {10},
        {8}, {2}, {8}, {6}, {4}
    };

    const float precision = 0.5f;
    unique_ptr<Data> data = make_unique<Data>(regression, inputData, expectedOutputs);
    data->setPrecision(precision);
    return data;
}

unique_ptr<Data> createRecurrentDataForAdditionTests(int numberOfData)
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;
    inputData.reserve(numberOfData);
    expectedOutputs.resize(numberOfData, {0});

    for (int i = 0; i < numberOfData; ++i)
    {
        auto r = snn::internal::Tools::randomBetween(0.0f, 0.25f);
        inputData.push_back({r});

        for (int j = 0; j < 4; ++j)
        {
            if (i + j < numberOfData)
                expectedOutputs[i + j][0] += r;
        }
    }

    const float precision = 0.05f;
    auto data = make_unique<Data>(regression, inputData, expectedOutputs, timeSeries, 1);
    data->setPrecision(precision);
    return data;
}
