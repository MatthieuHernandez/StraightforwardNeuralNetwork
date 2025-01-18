#include <algorithm>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

class NumericLimitTests : public testing::Test
{
    protected:
        static void createData(int sizeOfTraining, int sizeOfTesting)
        {
            vector2D<float> trainingExpectedOutputs;
            trainingExpectedOutputs.resize(sizeOfTraining);
            ranges::generate(trainingExpectedOutputs, [] { return vector<float>{tools::randomBetween(0.0f, 1.0f)}; });

            vector2D<float> trainingInputData;
            trainingInputData.resize(sizeOfTraining);
            ranges::generate(trainingInputData,
                             []
                             {
                                 return vector<float>{tools::randomBetween(0.0f, 1.0f),
                                                      tools::randomBetween(0.0f, 1.0f),
                                                      tools::randomBetween(0.0f, 1.0f)};
                             });

            vector2D<float> testingExpectedOutputs;
            testingExpectedOutputs.resize(sizeOfTesting);
            ranges::generate(testingExpectedOutputs, [] { return vector<float>{tools::randomBetween(0.0f, 1.0f)}; });

            vector2D<float> testingInputData(sizeOfTesting);
            testingInputData.resize(sizeOfTesting);
            ranges::generate(testingInputData,
                             []
                             {
                                 return vector<float>{tools::randomBetween(0.0f, 1.0f),
                                                      tools::randomBetween(0.0f, 1.0f),
                                                      tools::randomBetween(0.0f, 1.0f)};
                             });

            data = make_unique<Data>(problem::regression, trainingInputData, trainingExpectedOutputs, testingInputData,
                                     testingExpectedOutputs);
            data->setPrecision(0.3f);
        }

        static void testNeuralNetwork(StraightforwardNeuralNetwork& nn)
        {
            nn.train(*data, 1_s || 0.2_acc);
            auto mae = nn.getMeanAbsoluteError();
            auto acc = nn.getGlobalClusteringRate();
            ASSERT_ACCURACY(acc, 0.2f);
            ASSERT_MAE(mae, 1.4f);
        }

        static void SetUpTestSuite() { createData(1000, 50); }

        static unique_ptr<Data> data;
};

unique_ptr<Data> NumericLimitTests::data = nullptr;

TEST_F(NumericLimitTests, WithSigmoid)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(8, activation::sigmoid), FullyConnected(1, activation::sigmoid)},
        StochasticGradientDescent(0.01f, 0.99f));
    testNeuralNetwork(neuralNetwork);
}

TEST_F(NumericLimitTests, WithTanh)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(8, activation::tanh), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.01f, 0.99f));
    testNeuralNetwork(neuralNetwork);
}
