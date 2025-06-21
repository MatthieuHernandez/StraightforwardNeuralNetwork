#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Mnist.hpp"

using namespace snn;

class MnistTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Mnist datasetTest("./resources/datasets/MNIST");
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> MnistTest::dataset = nullptr;

TEST_F(MnistTest, loadData)
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

TEST_F(MnistTest, simplierNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1, 28, 28), MaxPooling(2), FullyConnected(10)},
                                               StochasticGradientDescent(0.02F, 0.1F));
    neuralNetwork.train(*dataset, 1_ep || 2_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.86F);
}

TEST_F(MnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(784), FullyConnected(150, activation::sigmoid, Dropout(0.1F)), FullyConnected(70), FullyConnected(10)},
        StochasticGradientDescent(0.05F, 0.85F));
    neuralNetwork.startTrainingAsync(*dataset);
    neuralNetwork.waitFor(1_ep || 15_s);
    neuralNetwork.stopTrainingAsync();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.91F);
}

TEST_F(MnistTest, feedforwardNeuralNetworkWithGRU)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(784),
                                                FullyConnected(100, activation::sigmoid, L2Regularization(1e-4F)),
                                                GruLayer(15), FullyConnected(10)});
    neuralNetwork.train(*dataset, 1_ep || 12_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90F);
}

TEST_F(MnistTest, locallyConnected1D)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(784), LocallyConnected(1, 7), FullyConnected(150, activation::sigmoid, L1Regularization(1e-5F)),
         FullyConnected(70), FullyConnected(10)},
        StochasticGradientDescent(0.1F));
    neuralNetwork.train(*dataset, 2_ep || 15_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.84F);
}

TEST_F(MnistTest, locallyConnected2D)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), LocallyConnected(2, 2), FullyConnected(150), FullyConnected(70), FullyConnected(10)},
        StochasticGradientDescent(0.1F, 0.7F));
    neuralNetwork.train(*dataset, 3_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.84F);
}

TEST_F(MnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(4, 3, activation::LeakyReLU), FullyConnected(10)},
        StochasticGradientDescent(0.0001F, 0.8F));
    neuralNetwork.train(*dataset, 1_ep || 15_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.87F);
}

TEST_F(MnistTest, DISABLED_multipleFilterConvolutionBetterThanOnce)
{
    StraightforwardNeuralNetwork nn1Filter(
        {Input(1, 28, 28), Convolution(1, 5, activation::sigmoid), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.0002F, 0.8F));
    nn1Filter.train(*dataset, 1_ep || 10_s);
    auto accuracy1Filter = nn1Filter.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy1Filter, 0.6F);

    StraightforwardNeuralNetwork nn8Filters(
        {Input(1, 28, 28), Convolution(4, 5, activation::sigmoid), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.0002F, 0.8F));
    nn8Filters.train(*dataset, 1_ep || 30_s);
    auto accuracy10Filters = nn8Filters.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy10Filters, 0.8F);

    EXPECT_GT(accuracy10Filters, accuracy1Filter);
}

TEST_F(MnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(12, 3, activation::GELU), MaxPooling(2), Convolution(24, 3, activation::GELU),
         MaxPooling(2), FullyConnected(92, activation::GELU), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.0005F, 0.8F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForMNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.99_acc);  // Achieved after 3 epochs, ~60 seconds each.

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98F);
}

TEST_F(MnistTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForMNIST.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*dataset);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();

    ASSERT_EQ(numberOfParameters, 83246);
    ASSERT_FLOAT_EQ(accuracy, 0.9900F);

    const std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       ./resources/BestNeuralNetworkForMNIST.snn
 Parameters: 83246
 Epochs:     8
 Trainnig:   480000
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 Convolution2D
                Input shape:  [1, 28, 28]
                Filters:      12
                Kernel size:  3x3
                Parameters:   120
                Activation:   GELU
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
                Activation:   GELU
                Output shape: [24, 11, 11]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [24, 11, 11]
                Kernel size:  2x2
                Output shape: [24, 6, 6]
------------------------------------------------------------
 FullyConnected
                Input shape:  [864]
                Neurons:      92
                Parameters:   79580
                Activation:   GELU
                Output shape: [92]
------------------------------------------------------------
 FullyConnected
                Input shape:  [92]
                Neurons:      10
                Parameters:   930
                Activation:   identity
                Output shape: [10]
                Optimizers:   Softmax
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 0.0005
                Momentum:      0.8
============================================================
)";
    const std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}

TEST_F(MnistTest, DISABLED_SaveFeatureMap)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(9, 3, activation::sigmoid, 0.0F, ErrorMultiplier(100.0F)),
         LocallyConnected(4, 2, activation::sigmoid, ErrorMultiplier(100.0F)), FullyConnected(10)},
        StochasticGradientDescent(0.005F, 0.1F));
    neuralNetwork.saveData2DAsBitmap("./bitmap/data", *dataset, 13);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before", *dataset, 13);
    neuralNetwork.train(*dataset, 1_ep);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90F);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/after", *dataset, 13);
}