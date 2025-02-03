#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Cifar10.hpp"
#include "ExtendedGTest.hpp"

using namespace snn;

class Cifar10Test : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Cifar10 datasetTest("./resources/datasets/CIFAR-10");
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> Cifar10Test::dataset = nullptr;

TEST_F(Cifar10Test, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 3072);
    ASSERT_EQ(dataset->numberOfLabels, 10);
    ASSERT_EQ(dataset->data.training.inputs.size(), 50000);
    ASSERT_EQ(dataset->data.training.labels.size(), 50000);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 10000);
    ASSERT_EQ(dataset->data.testing.labels.size(), 10000);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(Cifar10Test, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3072), FullyConnected(100), FullyConnected(40), FullyConnected(10, activation::identity, Softmax())});
    neuralNetwork.train(*dataset, 1_ep || 45_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.26F);
}

TEST_F(Cifar10Test, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3, 32, 32), Convolution(16, 3, activation::ReLU), MaxPooling(2), Convolution(32, 3, activation::ReLU),
         MaxPooling(2), FullyConnected(128), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.001F, 0.8F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForCIFAR-10.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.62_acc || 100_ep);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20F);
}

TEST_F(Cifar10Test, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForCIFAR-10.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*dataset);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 207210);
    ASSERT_FLOAT_EQ(accuracy, 0.6196F);  // Reach after 55 epochs of 770 sec.

    std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       BestNeuralNetworkForCIFAR-10.snn
 Parameters: 207210
 Epochs:     1
 Trainnig:   0
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 Convolution2D
                Input shape:  [3, 32, 32]
                Filters:      16
                Kernel size:  3x3
                Parameters:   448
                Activation:   ReLU
                Output shape: [16, 30, 30]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [16, 30, 30]
                Kernel size:  2x2
                Output shape: [16, 15, 15]
------------------------------------------------------------
 Convolution2D
                Input shape:  [16, 15, 15]
                Filters:      32
                Kernel size:  3x3
                Parameters:   4640
                Activation:   ReLU
                Output shape: [32, 13, 13]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [32, 13, 13]
                Kernel size:  2x2
                Output shape: [32, 7, 7]
------------------------------------------------------------
 FullyConnected
                Input shape:  [1568]
                Neurons:      128
                Parameters:   200832
                Activation:   sigmoid
                Output shape: [128]
------------------------------------------------------------
 FullyConnected
                Input shape:  [128]
                Neurons:      10
                Parameters:   1290
                Activation:   identity
                Output shape: [10]
                Optimizers:   Softmax
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 0.001
                Momentum:      0.8
============================================================
)";
    std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}

TEST_F(Cifar10Test, DISABLED_SaveFeatureMap)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForCIFAR-10.snn");

    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *dataset, 14);
    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *dataset, 15);
    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *dataset, 16);

    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_13", *dataset, 13);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_14", *dataset, 14);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_15", *dataset, 15);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_15", *dataset, 16);
}