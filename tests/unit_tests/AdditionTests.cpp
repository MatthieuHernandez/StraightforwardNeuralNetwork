#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

static auto createDataForAdditionTests() -> std::unique_ptr<Dataset>;
static auto createRecurrentDataForAdditionTests(int numberOfData, int numberOfRecurrences, float precision)
    -> std::unique_ptr<Dataset>;
static void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn);

TEST(Addition, WithMPL)
{
    auto dataset = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2), FullyConnected(16, activation::sigmoid), FullyConnected(1, activation::identity)},
        StochasticGradientDescent(0.02F));
    neuralNetwork.train(*dataset, 1.0_acc || 1_s, 3, 4);
    testNeuralNetworkForAddition(neuralNetwork);
}

TEST(Addition, WithCNN)
{
    auto dataset = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2), Convolution(6, 1, activation::sigmoid), FullyConnected(1, activation::identity)},
        StochasticGradientDescent(0.004F));

    neuralNetwork.train(*dataset, 1.0_acc || 2_s);
    testNeuralNetworkForAddition(neuralNetwork);
}

TEST(Addition, WithLCNN)
{
    auto dataset = createDataForAdditionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2), LocallyConnected(6, 1, activation::sigmoid), FullyConnected(1, activation::identity)},
        StochasticGradientDescent(0.01F));

    neuralNetwork.train(*dataset, 1.0_acc || 5_s);
    testNeuralNetworkForAddition(neuralNetwork);
}

TEST(Addition, WithRNN)
{
    auto dataset = createRecurrentDataForAdditionTests(100, 3, 0.3F);
    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(12), Recurrence(5), FullyConnected(1)},
                                               StochasticGradientDescent(0.02F, 0.4F));

    neuralNetwork.train(*dataset, 1.0_acc || 3_s, 1, 2);
    testNeuralNetworkForAddition(neuralNetwork);
}

TEST(Addition, WithGRU)
{
    auto dataset = createRecurrentDataForAdditionTests(100, 3, 0.3F);
    StraightforwardNeuralNetwork neuralNetwork({Input(1), GruLayer(16), GruLayer(12), FullyConnected(1)},
                                               StochasticGradientDescent(0.005F, 0.95F));

    neuralNetwork.train(*dataset, 1.0_acc || 3_s, 1, 3);
    testNeuralNetworkForAddition(neuralNetwork);
}

void testNeuralNetworkForAddition(StraightforwardNeuralNetwork& nn)
{
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0F);
    ASSERT_MAE(mae, 0.4F);
}

auto createDataForAdditionTests() -> std::unique_ptr<Dataset>
{
    vector2D<float> inputData = {{3, 5}, {5, 4}, {4, 2}, {2, 0}, {0, 2}, {2, 4}, {4, 1}, {1, 4}, {4, 3},
                                 {3, 0}, {0, 0}, {0, 4}, {4, 3}, {3, 2}, {2, 1}, {1, 2}, {2, 0}, {0, 1},
                                 {1, 2}, {5, 5}, {5, 3}, {1, 1}, {4, 4}, {3, 3}, {2, 2}};
    vector2D<float> expectedOutputs = {{8}, {9}, {6}, {2}, {2}, {6}, {5},  {5}, {7}, {3}, {0}, {4}, {7},
                                       {5}, {3}, {3}, {2}, {1}, {3}, {10}, {8}, {2}, {8}, {6}, {4}};

    const float precision = 0.4F;
    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs);
    dataset->setPrecision(precision);
    return dataset;
}

auto createRecurrentDataForAdditionTests(int numberOfData, int numberOfRecurrences, float precision)
    -> std::unique_ptr<Dataset>
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;
    inputData.reserve(numberOfData);
    expectedOutputs.resize(numberOfData, {0});

    for (int i = 0; i < numberOfData; ++i)
    {
        auto rnd = tools::randomBetween(0.0F, 1.0F / static_cast<float>(numberOfRecurrences + 1));
        inputData.push_back({rnd});

        for (int j = 0; j < numberOfRecurrences + 1; ++j)
        {
            if (i + j < numberOfData)
            {
                expectedOutputs[(i + j)][0] += rnd;
            }
        }
    }

    auto dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs, nature::timeSeries,
                                             numberOfRecurrences);
    dataset->setPrecision(precision);
    return dataset;
}
