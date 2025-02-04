#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

auto createDataForOptimizerTests(int numberOfData, int sizeOfData) -> std::unique_ptr<Dataset>;

TEST(Optimizer, FindRightValueIn20)
{
    std::unique_ptr<Dataset> dataset = createDataForOptimizerTests(1000, 20);
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(20), FullyConnected(8, activation::tanh), FullyConnected(1, activation::sigmoid)},
        StochasticGradientDescent(0.05F, 0.9F));

    neuralNetwork.train(*dataset, 0.99_acc || 2_s);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    auto acc = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 0.99F);
    ASSERT_MAE(mae, 0.3F);
}

auto createDataForOptimizerTests(int numberOfData, int sizeOfData) -> std::unique_ptr<Dataset>
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;

    inputData.reserve(numberOfData);
    expectedOutputs.reserve(numberOfData);
    for (int i = 0; i < numberOfData; ++i)
    {
        inputData.emplace_back();
        inputData.back().reserve(sizeOfData);
        for (int j = 0; j < sizeOfData; ++j)
        {
            const float rand = tools::randomBetween(-1.0F, 1.0F);
            inputData.back().push_back(rand);
        }
        if (inputData[i][0] > 0)
        {
            expectedOutputs.push_back({1.0F});
        }
        else
        {
            expectedOutputs.push_back({0.0F});
        }
    }
    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>(problem::regression, inputData, expectedOutputs);
    dataset->setPrecision(0.3F);
    return dataset;
}
