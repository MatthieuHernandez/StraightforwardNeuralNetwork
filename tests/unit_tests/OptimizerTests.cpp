#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

auto createDataForOptimizerTests(int numberOfData, int sizeOfData) -> std::unique_ptr<Data>;

TEST(Optimizer, FindRightValueIn20)
{
    std::unique_ptr<Data> data = createDataForOptimizerTests(1000, 20);
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(20), FullyConnected(8, activation::tanh), FullyConnected(1, activation::sigmoid)},
        StochasticGradientDescent(0.05F, 0.9F));

    neuralNetwork.train(*data, 0.99_acc || 2_s);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    auto acc = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 0.99F);
    ASSERT_MAE(mae, 0.3F);
}

auto createDataForOptimizerTests(int numberOfData, int sizeOfData) -> std::unique_ptr<Data>
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;

    inputData.reserve(numberOfData);
    expectedOutputs.reserve(numberOfData);
    for (int i = 0; i < numberOfData; ++i)
    {
        inputData.push_back(std::vector<float>());
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
    std::unique_ptr<Data> data = std::make_unique<Data>(problem::regression, inputData, expectedOutputs);
    data->setPrecision(0.3F);
    return data;
}
