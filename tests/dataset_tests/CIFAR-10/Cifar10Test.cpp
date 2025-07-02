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
        {Input(3, 32, 32), Convolution(24, 3, activation::ReLU), MaxPooling(2), Convolution(48, 3, activation::ReLU),
         MaxPooling(2), FullyConnected(150, activation::ReLU), FullyConnected(10, activation::identity)},
        StochasticGradientDescent(2e-3F, 0.85F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForCIFAR-10.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.80_acc || 100_ep);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.60F);
}

TEST_F(Cifar10Test, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForCIFAR-10.snn");
    const std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       ./resources/BestNeuralNetworkForCIFAR-10.snn
 Parameters: 473548
 Epochs:     4
 Trainnig:   200000
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 Convolution2D
                Input shape:  [3, 32, 32]
                Filters:      24
                Kernel size:  3x3
                Parameters:   672
                Activation:   ReLU
                Output shape: [24, 32, 32]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [24, 32, 32]
                Kernel size:  2x2
                Output shape: [24, 16, 16]
------------------------------------------------------------
 Convolution2D
                Input shape:  [24, 16, 16]
                Filters:      48
                Kernel size:  3x3
                Parameters:   10416
                Activation:   ReLU
                Output shape: [48, 16, 16]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [48, 16, 16]
                Kernel size:  2x2
                Output shape: [48, 8, 8]
------------------------------------------------------------
 FullyConnected
                Input shape:  [3072]
                Neurons:      150
                Parameters:   460950
                Activation:   ReLU
                Output shape: [150]
------------------------------------------------------------
 FullyConnected
                Input shape:  [150]
                Neurons:      10
                Parameters:   1510
                Activation:   identity
                Output shape: [10]
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 0.002
                Momentum:      0.85
============================================================
)";
    const std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
    const auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    ASSERT_EQ(numberOfParameters, 473548);
    neuralNetwork.evaluate(*dataset);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_FLOAT_EQ(accuracy, 0.6623F);  // Achieved after 4 epochs, ~630 seconds each.
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