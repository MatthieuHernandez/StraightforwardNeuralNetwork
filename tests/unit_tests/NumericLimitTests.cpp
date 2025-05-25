#include <algorithm>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

class NumericLimitTests : public testing::Test
{
    protected:
        static void createData(int sizeOfTraining, int sizeOfTesting)
        {
            vector2D<float> trainingExpectedOutputs;
            trainingExpectedOutputs.resize(sizeOfTraining);
            std::ranges::generate(trainingExpectedOutputs,
                                  [] { return std::vector<float>{tools::randomBetween(0.0F, 1.0F)}; });

            vector2D<float> trainingInputData;
            trainingInputData.resize(sizeOfTraining);
            std::ranges::generate(trainingInputData,
                                  []
                                  {
                                      return std::vector<float>{tools::randomBetween(0.0F, 1.0F),
                                                                tools::randomBetween(0.0F, 1.0F),
                                                                tools::randomBetween(0.0F, 1.0F)};
                                  });

            vector2D<float> testingExpectedOutputs;
            testingExpectedOutputs.resize(sizeOfTesting);
            std::ranges::generate(testingExpectedOutputs,
                                  [] { return std::vector<float>{tools::randomBetween(0.0F, 1.0F)}; });

            vector2D<float> testingInputData(sizeOfTesting);
            testingInputData.resize(sizeOfTesting);
            std::ranges::generate(testingInputData,
                                  []
                                  {
                                      return std::vector<float>{tools::randomBetween(0.0F, 1.0F),
                                                                tools::randomBetween(0.0F, 1.0F),
                                                                tools::randomBetween(0.0F, 1.0F)};
                                  });

            dataset = std::make_unique<Dataset>(problem::regression, trainingInputData, trainingExpectedOutputs,
                                                testingInputData, testingExpectedOutputs);
            dataset->setPrecision(0.3F);
        }

        static void testNeuralNetwork(StraightforwardNeuralNetwork& neuralNetwork)
        {
            neuralNetwork.train(*dataset, 1_s || 0.2_acc);
            auto mae = neuralNetwork.getMeanAbsoluteError();
            auto acc = neuralNetwork.getGlobalClusteringRate();
            ASSERT_ACCURACY(acc, 0.2F);
            ASSERT_MAE(mae, 1.4F);
        }

        static void SetUpTestSuite() { createData(1000, 50); }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> NumericLimitTests::dataset = nullptr;

TEST_F(NumericLimitTests, WithSigmoid)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(10, activation::sigmoid), FullyConnected(1, activation::sigmoid)},
        StochasticGradientDescent(0.01F, 0.99F));
    testNeuralNetwork(neuralNetwork);
}

TEST_F(NumericLimitTests, WithTanh)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3), FullyConnected(10, activation::tanh), FullyConnected(1, activation::tanh)},
        StochasticGradientDescent(0.01F, 0.99F));
    testNeuralNetwork(neuralNetwork);
}
