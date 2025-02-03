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
        {Input(784), FullyConnected(150), FullyConnected(70), FullyConnected(10)});
    neuralNetwork.train(*dataset, 1_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.76F);
}

TEST_F(FashionMnistTest, convolutionNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1, 28, 28), Convolution(6, 3, activation::ReLU),
                                                Convolution(4, 5, activation::ReLU), FullyConnected(10)},
                                               StochasticGradientDescent(0.0002F, 0.80F));
    neuralNetwork.train(*dataset, 1_ep || 1_min);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70F);
}

TEST_F(FashionMnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(12, 3, activation::ReLU), MaxPooling(2), Convolution(24, 3, activation::ReLU),
         FullyConnected(92), FullyConnected(10)},
        StochasticGradientDescent(0.005F, 0.93F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForFashion-MNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.92_acc);  // Reach after 109 epochs of 2 sec.

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20F);
}

TEST_F(FashionMnistTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForFashion-MNIST.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*dataset);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 270926);
    ASSERT_FLOAT_EQ(accuracy, 0.8965F);

    std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       BestNeuralNetworkForFashion-MNIST.snn
 Parameters: 270926
 Epochs:     15
 Trainnig:   0
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
                Output shape: [12, 26, 26]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [12, 26, 26]
                Kernel size:  2x2
                Output shape: [12, 13, 13]
------------------------------------------------------------
 Convolution2D
                Input shape:  [12, 13, 13]
                Filters:      24
                Kernel size:  3x3
                Parameters:   2616
                Activation:   ReLU
                Output shape: [24, 11, 11]
------------------------------------------------------------
 FullyConnected
                Input shape:  [2904]
                Neurons:      92
                Parameters:   267260
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
                Learning rate: 0.005
                Momentum:      0.93
============================================================
)";
    std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}
