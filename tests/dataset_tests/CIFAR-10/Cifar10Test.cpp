#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "Cifar10.hpp"
#include "ExtendedGTest.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class Cifar10Test : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Cifar10 dataset("./resources/datasets/CIFAR-10");
            data = move(dataset.data);
        }

        void SetUp() final { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static std::unique_ptr<Data> data;
};

unique_ptr<Data> Cifar10Test::data = nullptr;

TEST_F(Cifar10Test, loadData)
{
    ASSERT_EQ(data->sizeOfData, 3072);
    ASSERT_EQ(data->numberOfLabels, 10);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 50000);
    ASSERT_EQ((int)data->sets[training].labels.size(), 50000);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), ErrorType::noError);
}

TEST_F(Cifar10Test, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(3072), FullyConnected(80), FullyConnected(30), FullyConnected(10, activation::identity, Softmax())});
    neuralNetwork.train(*data, 1_ep || 45_s);
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
    neuralNetwork.train(*data, 0.62_acc || 100_ep);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20F);
}

TEST_F(Cifar10Test, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForCIFAR-10.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 207210);
    ASSERT_FLOAT_EQ(accuracy, 0.6196F);  // Reach after 55 epochs of 770 sec.

    string expectedSummary =
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
    string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}

TEST_F(Cifar10Test, DISABLED_SaveFeatureMap)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForCIFAR-10.snn");

    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *data, 13);
    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *data, 14);
    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *data, 15);
    neuralNetwork.saveData2DAsBitmap("./bitmap/before", *data, 16);

    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_13", *data, 13);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_14", *data, 14);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_15", *data, 15);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before_15", *data, 16);
}