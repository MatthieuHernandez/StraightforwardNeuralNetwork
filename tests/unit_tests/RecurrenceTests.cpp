#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

static void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Dataset& d);

TEST(Recurrence, RepeatInput)
{
    vector2D<float> inputData = {{0.0F}, {-1.0F}, {-0.8F}, {-0.5F}, {-0.2F}, {0.0F}, {0.3F}, {0.5F}, {0.7F}, {1.0F}};
    vector2D<float> expectedOutputs = {{0.0F}, {-1.0F}, {-0.8F}, {-0.5F}, {-0.2F},
                                       {0.0F}, {0.3F},  {0.5F},  {0.7F},  {1.0F}};
    auto dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    dataset->setPrecision(0.15F);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), Recurrence(12), FullyConnected(6), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.003F, 0.97F));
    testNeuralNetworkForRecurrence(neuralNetwork, *dataset);
}

TEST(Recurrence, RepeatLastInput)
{
    size_t datasetSize = 1000;
    std::uniform_int_distribution<int> dist(-1, 1);
    vector2D<float> inputData;
    inputData.reserve(datasetSize);
    for (size_t i = 0; i < datasetSize; ++i)
    {
        inputData.push_back({static_cast<float>(dist(tools::Rng()))});
    }
    vector2D<float> expectedOutputs = {{0}};
    expectedOutputs.reserve(datasetSize);
    expectedOutputs.insert(expectedOutputs.end(), inputData.cbegin(), inputData.cend() - 1);

    auto dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 1);
    dataset->setPrecision(0.5F);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(30), Recurrence(1, activation::tanh)},
                                               StochasticGradientDescent(0.0001F, 0.7F));
    testNeuralNetworkForRecurrence(neuralNetwork, *dataset);
}

// a simple recurrent neural network can't solve this problem
TEST(Recurrence, RepeatLastLastInput)
{
    size_t datasetSize = 100;
    std::uniform_int_distribution<int> dist(-1, 1);
    vector2D<float> inputData;
    inputData.reserve(datasetSize);
    for (size_t i = 0; i < datasetSize; ++i)
    {
        inputData.push_back({static_cast<float>(dist(tools::Rng()))});
    }
    vector2D<float> expectedOutputs = {{0, 0}};
    expectedOutputs.reserve(datasetSize);
    expectedOutputs.insert(expectedOutputs.end(), inputData.cbegin(), inputData.cend() - 2);

    auto dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs, nature::timeSeries, 2);
    dataset->setPrecision(0.3F);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), GruLayer(50), FullyConnected(1, activation::tanh)},
                                               StochasticGradientDescent(0.0005F, 0.7F));

    testNeuralNetworkForRecurrence(neuralNetwork, *dataset);
}

inline static void testNeuralNetworkForRecurrence(StraightforwardNeuralNetwork& nn, Dataset& d)
{
    nn.train(d, 0.9_acc || 4_s, 1, 2);
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 0.9);
    ASSERT_MAE(mae, 0.5);
}
