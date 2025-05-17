#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Affnist.hpp"
#include "ExtendedGTest.hpp"

using namespace snn;

class AffnistTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Affnist datasetTest("./resources/datasets/affNIST");
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> AffnistTest::dataset = nullptr;

TEST_F(AffnistTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 1600);
    ASSERT_EQ(dataset->numberOfLabels, 10);
    ASSERT_EQ(dataset->data.training.inputs.size(), 60000);
    ASSERT_EQ(dataset->data.training.labels.size(), 60000);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 10000);
    ASSERT_EQ(dataset->data.testing.labels.size(), 10000);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(AffnistTest, simplierNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1, 40, 40), MaxPooling(2), FullyConnected(10)},
                                               StochasticGradientDescent(0.02F, 0.1F));
    neuralNetwork.train(*dataset, 5_ep);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.86F);
}

TEST_F(AffnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1600), FullyConnected(200, activation::ReLU),
                                                FullyConnected(100, activation::ReLU), FullyConnected(10)});
    neuralNetwork.startTrainingAsync(*dataset);
    neuralNetwork.waitFor(20_ep);
    neuralNetwork.stopTrainingAsync();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.91F);
}

TEST_F(AffnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 40, 40), Convolution(8, 3, activation::ReLU), MaxPooling(2), Convolution(24, 3, activation::ReLU),
         Convolution(72, 3, activation::ReLU), MaxPooling(2), FullyConnected(10)});
    neuralNetwork.train(*dataset, 5_ep);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.87F);
}