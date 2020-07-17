#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForOptimisezTests(int numberOfData, int sizeOfData);

TEST(Optimizer, FindRightValueIn100)
{
    unique_ptr<Data> data = createDataForOptimisezTests(1000, 100);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        FullyConnected(1, sigmoid)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1.0_acc || 10_s);
    neuralNetwork.stopTraining();
    auto mae = neuralNetwork.getMeanAbsoluteError();
    auto acc = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 1.0);
    ASSERT_MAE(mae, 0.6);
}

unique_ptr<Data> createDataForOptimisezTests(int numberOfData, int sizeOfData)
{
    vector2D<float> inputData;
    vector2D<float> expectedOutputs;

    inputData.reserve(numberOfData);
    expectedOutputs.reserve(sizeOfData);
    for(int i; i < numberOfData; ++i)
    {
        inputData.push_back({});
        for(int j; j < sizeOfData; ++i)
        {
            const auto rand = snn::internal::Tools::randomBetween(-1, 1);
            inputData.back().push_back(rand);
        }
        if(inputData[i][0] > 0)
            expectedOutputs.push_back({1});
        else
            expectedOutputs.push_back({0});
    }

    const float precision = 0.5f;
    unique_ptr<Data> data = make_unique<Data>(regression, inputData, expectedOutputs);
    data->setPrecision(precision);
    return data;
}
