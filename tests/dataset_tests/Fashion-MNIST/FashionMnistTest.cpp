#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "FashionMnist.hpp"

using namespace snn;

class FashionMnistTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            FashionMnist datasetTest("./resources/datasets/Fashion-MNIST");
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> FashionMnistTest::dataset = nullptr;

TEST_F(FashionMnistTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 784);
    ASSERT_EQ(dataset->numberOfLabels, 10);
    ASSERT_EQ(dataset->data.training.inputs.size(), 60000);
    ASSERT_EQ(dataset->data.training.labels.size(), 60000);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 10000);
    ASSERT_EQ(dataset->data.testing.labels.size(), 10000);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(FashionMnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(784), FullyConnected(150), FullyConnected(70), FullyConnected(10)},
        StochasticGradientDescent(0.05F, 0.9F));
    neuralNetwork.train(*dataset, 1_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.76F);
}

TEST_F(FashionMnistTest, convolutionNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), MaxPooling(2), Convolution(10, 3, activation::LeakyReLU),
         Convolution(20, 5, activation::LeakyReLU), FullyConnected(10)},
        StochasticGradientDescent(0.0008F, 0.92F));
    neuralNetwork.train(*dataset, 1_ep || 1_min);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70F);
}

TEST_F(FashionMnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(12, 3, activation::ReLU), MaxPooling(2), Convolution(24, 3, activation::ReLU),
         FullyConnected(92), FullyConnected(10, activation::sigmoid)},
        StochasticGradientDescent(0.001F, 0.70F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForFashion-MNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.92_acc);  // Achieved after 13 epochs, ~120 seconds each.

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.85F);
}

TEST_F(FashionMnistTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForFashion-MNIST.snn");
    const std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       ./resources/BestNeuralNetworkForFashion-MNIST.snn
 Parameters: 436526
 Epochs:     13
 Trainnig:   780000
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 Convolution2D
                Input shape:  [1, 28, 28]
                Filters:      12
                Kernel size:  3x3
                Parameters:   120
                Activation:   ReLU
                Output shape: [12, 28, 28]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [12, 28, 28]
                Kernel size:  2x2
                Output shape: [12, 14, 14]
------------------------------------------------------------
 Convolution2D
                Input shape:  [12, 14, 14]
                Filters:      24
                Kernel size:  3x3
                Parameters:   2616
                Activation:   ReLU
                Output shape: [24, 14, 14]
------------------------------------------------------------
 FullyConnected
                Input shape:  [4704]
                Neurons:      92
                Parameters:   432860
                Activation:   sigmoid
                Output shape: [92]
------------------------------------------------------------
 FullyConnected
                Input shape:  [92]
                Neurons:      10
                Parameters:   930
                Activation:   sigmoid
                Output shape: [10]
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 0.001
                Momentum:      0.7
============================================================
)";
    const std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    ASSERT_EQ(numberOfParameters, 436526);
    neuralNetwork.evaluate(*dataset);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_FLOAT_EQ(accuracy, 0.8961F);
}
