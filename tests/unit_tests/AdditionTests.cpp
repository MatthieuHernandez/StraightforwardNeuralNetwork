#include <cstddef>
#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForAdditionTests();
unique_ptr<Data> createRecurrentDataForAdditionTests(int numberOfData, int numberOfRecurrences, float precision);
void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d);

TEST(Addition, WithMPL)
{
    unique_ptr<Data> data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        FullyConnected(6, activation::sigmoid),
        FullyConnected(1, activation::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithCNN)
{
    auto data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        Convolution(6, 1, activation::sigmoid),
        FullyConnected(1, activation::identity)
    }, 
        StochasticGradientDescent(0.01f));

    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithLCNN)
{
    auto data = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        LocallyConnected(6, 1, activation::sigmoid),
        FullyConnected(1, activation::identity)
    });
    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithRNN)
{
    auto data = createRecurrentDataForAdditionTests(100, 3, 0.3f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12),
        Recurrence(5),
        FullyConnected(1)
    }, 
        StochasticGradientDescent(0.01f, 0.4f));

    testNeuralNetworkForAddition(neuralNetwork, *data);
}

TEST(Addition, WithGRU)
{
    auto data = createRecurrentDataForAdditionTests(100, 3, 0.3f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        GruLayer(15),
         GruLayer(5),
        FullyConnected(1)
    }, 
        StochasticGradientDescent(0.01f, 0.4f));

    testNeuralNetworkForAddition(neuralNetwork, *data);
}

void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.train(d, 1.0_acc || 8_s);
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
    unique_ptr<Data> data = make_unique<Data>(problem::regression, inputData, expectedOutputs);
    data->setPrecision(precision);
    return data;
}

unique_ptr<Data> createRecurrentDataForAdditionTests(int numberOfData, int numberOfRecurrences, float precision)
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;
    inputData.reserve(numberOfData);
    expectedOutputs.resize(numberOfData, {0});

    for (int i = 0; i < numberOfData; ++i)
    {
        auto r = internal::Tools::randomBetween(0.0f, 1.0f/(numberOfRecurrences+1));
        inputData.push_back({r});

        for (int j = 0; j < numberOfRecurrences+1; ++j)
        {
            if (i + j < numberOfData)
                expectedOutputs[(int)(i + j)][0] += r;
        }
    }

    auto data = make_unique<Data>(problem::regression, inputData, expectedOutputs, nature::timeSeries, numberOfRecurrences);
    data->setPrecision(precision);
    return data;
}
