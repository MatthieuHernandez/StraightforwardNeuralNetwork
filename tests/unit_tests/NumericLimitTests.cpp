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
            ranges::generate(trainingExpectedOutputs,
                             [] { return std::vector<float>{tools::randomBetween(0.0F, 1.0F)}; });

            vector2D<float> trainingInputData;
            trainingInputData.resize(sizeOfTraining);
            ranges::generate(trainingInputData,
                             []
                             {
                                 return std::vector<float>{tools::randomBetween(0.0F, 1.0F),
                                                           tools::randomBetween(0.0F, 1.0F),
                                                           tools::randomBetween(0.0F, 1.0F)};
                             });

            vector2D<float> testingExpectedOutputs;
            testingExpectedOutputs.resize(sizeOfTesting);
            ranges::generate(testingExpectedOutputs,
                             [] { return std::vector<float>{tools::randomBetween(0.0F, 1.0F)}; });

            vector2D<float> testingInputData(sizeOfTesting);
            testingInputData.resize(sizeOfTesting);
            ranges::generate(testingInputData,
                             []
                             {
                                 return std::vector<float>{tools::randomBetween(0.0F, 1.0F),
                                                           tools::randomBetween(0.0F, 1.0F),
                                                           tools::randomBetween(0.0F, 1.0F)};
                             });

            data = std::make_unique<Data>(problem::regression, trainingInputData, trainingExpectedOutputs,
                                          testingInputData, testingExpectedOutputs);
            data->setPrecision(0.3F);
        }

        static void testNeuralNetwork(StraightforwardNeuralNetwork& nn)
        {
            nn.train(*data, 1_s || 0.2_acc);
            auto mae = nn.getMeanAbsoluteError();
            auto acc = nn.getGlobalClusteringRate();
            ASSERT_ACCURACY(acc, 0.2F);
            ASSERT_MAE(mae, 1.4F);
        }

        static void SetUpTestSuite() { createData(1000, 50); }

        static std::unique_ptr<Data> data;
};

unique_ptr<Data> NumericLimitTests::data = nullptr;

TEST_F(NumericLimitTests, WithSigmoid)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(8, activation::sigmoid), FullyConnected(1, activation::sigmoid)},
        StochasticGradientDescent(0.01F, 0.99F));
    testNeuralNetwork(neuralNetwork);
}

TEST_F(NumericLimitTests, WithTanh)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(8, activation::tanh), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.01F, 0.99F));
    testNeuralNetwork(neuralNetwork);
}
