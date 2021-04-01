#include <cstddef>
#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

class NumericLimitTests : public testing::Test
{
protected:
    static void createData(int sizeOfTraining, int sizeOfTesting)
    {
        vector2D<float> trainingExpectedOutputs;
        trainingExpectedOutputs.resize(sizeOfTraining);
        std::generate(trainingExpectedOutputs.begin(), trainingExpectedOutputs.end(), []
        {
            return vector<float>{tools::randomBetween(0.0f, 1.0f)};
        });

        vector2D<float> trainingInputData;
        trainingInputData.resize(sizeOfTraining);
        std::generate(trainingInputData.begin(), trainingInputData.end(), []
        {
            return vector<float>
            {
                tools::randomBetween(0.0f, 1.0f),
                tools::randomBetween(0.0f, 1.0f),
                tools::randomBetween(0.0f, 1.0f)
            };
        });

        vector2D<float> testingExpectedOutputs;
        testingExpectedOutputs.resize(sizeOfTesting);
        std::generate(testingExpectedOutputs.begin(), testingExpectedOutputs.end(), []
        {
            return vector<float>{tools::randomBetween(0.0f, 1.0f)};
        });

        vector2D<float> testingInputData(sizeOfTesting);
        testingInputData.resize(sizeOfTesting);
        std::generate(testingInputData.begin(), testingInputData.end(), []
        {
            return vector<float>
            {
                tools::randomBetween(0.0f, 1.0f),
                tools::randomBetween(0.0f, 1.0f),
                tools::randomBetween(0.0f, 1.0f)
            };
        });

        data = make_unique<Data>(problem::regression,
                                 trainingInputData,
                                 trainingExpectedOutputs,
                                 testingInputData,
                                 testingExpectedOutputs);
        data->setPrecision(0.5f);
    }

    static void testNeuralNetwork(StraightforwardNeuralNetwork& nn)
    {
        nn.train(*data, 1_s);
        auto mae = nn.getMeanAbsoluteError();
        auto acc = nn.getGlobalClusteringRate();
        ASSERT_ACCURACY(acc, 0.1f);
        ASSERT_MAE(mae, 1.5f);
    }

    static void SetUpTestSuite()
    {
        createData(1000, 50);
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> NumericLimitTests::data = nullptr;

TEST_F(NumericLimitTests, WithSigmoid)
{
    StraightforwardNeuralNetwork neuralNetwork({
                                                   Input(3),
                                                   FullyConnected(5, activation::sigmoid),
                                                   FullyConnected(1, activation::sigmoid)
                                               },
                                               StochasticGradientDescent(0.9999f, 0.9999f));
    testNeuralNetwork(neuralNetwork);
}

TEST_F(NumericLimitTests, WithTanh)
{
    StraightforwardNeuralNetwork neuralNetwork({
                                                   Input(3),
                                                   FullyConnected(5, activation::tanh),
                                                   FullyConnected(1, activation::tanh)
                                               },
                                               StochasticGradientDescent(0.9999f, 0.9999f));
    testNeuralNetwork(neuralNetwork);
}