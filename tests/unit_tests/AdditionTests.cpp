#include <gtest/gtest.h>

#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "custom_dataset/AdditionDataset.hpp"

using namespace snn;

TEST(Addition, WithMPL)
{
    // This model learn to sum 10 float numbers between 0 and 10 with a precision of 0.4.
    auto dataset = addition::createNonTemporalDataset(10000, 10, 0.4F);
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(10), FullyConnected(30, activation::ReLU), FullyConnected(1, activation::identity)},
        StochasticGradientDescent(1e-4F, 0.9F));
    addition::trainAndTest(neuralNetwork, dataset);
}

TEST(Addition, WithRNN)
{
    auto dataset = addition::createTimeSeriesDataset(10000, 10, 0.4F);
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), Recurrence(30, activation::ReLU), FullyConnected(1, activation::identity)},
        StochasticGradientDescent(1.0e-4F, 0.9F));
    addition::trainAndTest(neuralNetwork, dataset);
}

/*
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
                                               StochasticGradientDescent(0.02F, 0.5F));

    neuralNetwork.train(*dataset, 1.0_acc || 3_s, 1, 2);
    testNeuralNetworkForAddition(neuralNetwork);
}

TEST(Addition, WithGRU)
{
    auto dataset = createRecurrentDataForAdditionTests(100, 3, 0.3F);
    StraightforwardNeuralNetwork neuralNetwork({Input(1), GruLayer(16), GruLayer(12), FullyConnected(1)},
                                               StochasticGradientDescent(0.005F, 0.96F));

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
*/
