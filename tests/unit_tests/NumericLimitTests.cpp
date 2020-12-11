#include <cstddef>
#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

unique_ptr<Data> createDataForNumericLimitTests(int sizeOfTraining, int sizeOfTesting, float output);
void testNeuralNetworkForNumericLimitTests(StraightforwardNeuralNetwork& nn, Data& d);

TEST(NumericLimit, AlwaysOneWithSigmoid)
{
    unique_ptr<Data> data = createDataForNumericLimitTests(5000, 100, 1.0f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3),
        FullyConnected(3, activation::sigmoid),
        FullyConnected(1, activation::sigmoid)
    });
    testNeuralNetworkForNumericLimitTests(neuralNetwork, *data);
}

TEST(NumericLimit, AlwaysZeroWithSigmoid)
{
    unique_ptr<Data> data = createDataForNumericLimitTests(5000, 100, 0.0f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3),
        FullyConnected(3, activation::sigmoid),
        FullyConnected(1, activation::sigmoid)
    });
    testNeuralNetworkForNumericLimitTests(neuralNetwork, *data);
}

TEST(NumericLimit, AlwaysOneWithTanh)
{
    unique_ptr<Data> data = createDataForNumericLimitTests(5000, 100, 1.0f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3),
        FullyConnected(3, activation::tanh),
        FullyConnected(1, activation::tanh)
    });
    testNeuralNetworkForNumericLimitTests(neuralNetwork, *data);
}

TEST(NumericLimit, AlwaysZeroWithTanh)
{
    unique_ptr<Data> data = createDataForNumericLimitTests(5000, 100, 0.0f);
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3),
        FullyConnected(3, activation::tanh),
        FullyConnected(1, activation::tanh)
    });
    testNeuralNetworkForNumericLimitTests(neuralNetwork, *data);
}

void testNeuralNetworkForNumericLimitTests(StraightforwardNeuralNetwork& nn, Data& d)
{
    nn.startTraining(d);
    nn.waitFor(10_s);
    nn.stopTraining();
    auto mae = nn.getMeanAbsoluteError();
    auto acc = nn.getGlobalClusteringRateMax();
    ASSERT_ACCURACY(acc, 1.0f);
    ASSERT_MAE(mae, 0.01f);
}

unique_ptr<Data> createDataForNumericLimitTests(int sizeOfTraining, int sizeOfTesting, float output)
{
    vector2D<float> trainingInputData;
    vector2D<float> trainingExpectedOutputs;
    trainingInputData.reserve(sizeOfTraining);
    trainingExpectedOutputs.resize(sizeOfTraining, {output});

    for (int i = 0; i < sizeOfTraining; ++i)
    {
        auto r1 = internal::Tools::randomBetween(0.0f, 1.0f);
        auto r2 = internal::Tools::randomBetween(0.0f, 1.0f);
        auto r3 = internal::Tools::randomBetween(0.0f, 1.0f);
        trainingInputData.push_back({r1, r2, r3});
    }

    vector2D<float> testingInputData(sizeOfTesting);
    std::generate(testingInputData.begin(), testingInputData.end(), [] { return vector<float>{0.0f}; });


    vector2D<float> testingExpectedOutputs;
    testingExpectedOutputs.resize(sizeOfTesting, {output});

    for (int i = 0; i < sizeOfTesting; ++i)
    {
       auto r1 = internal::Tools::randomBetween(0.0f, 1.0f);
        auto r2 = internal::Tools::randomBetween(0.0f, 1.0f);
        auto r3 = internal::Tools::randomBetween(0.0f, 1.0f);
        trainingInputData.push_back({r1, r2, r3});
    }

    auto data = make_unique<Data>(problem::regression,
                                  trainingInputData,
                                  trainingExpectedOutputs,
                                  testingInputData,
                                  testingExpectedOutputs);
    return data;
}