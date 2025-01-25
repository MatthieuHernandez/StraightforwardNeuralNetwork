#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Mnist.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class MnistTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Mnist dataset("./resources/datasets/MNIST");
            data = move(dataset.data);
        }

        void SetUp() override { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static unique_ptr<Data> data;
};

unique_ptr<Data> MnistTest::data = nullptr;

TEST_F(MnistTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 784);
    ASSERT_EQ(data->numberOfLabels, 10);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 60000);
    ASSERT_EQ((int)data->sets[training].labels.size(), 60000);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), ErrorType::noError);
}

TEST_F(MnistTest, simplierNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1, 28, 28), MaxPooling(2), FullyConnected(10)},
                                               StochasticGradientDescent(0.02F, 0.1F));
    neuralNetwork.train(*data, 1_ep || 2_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.86F);
}

TEST_F(MnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(784), FullyConnected(150, activation::sigmoid, Dropout(0.1F)), FullyConnected(70), FullyConnected(10)});
    neuralNetwork.startTrainingAsync(*data);
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
    neuralNetwork.train(*data, 1_ep || 12_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90F);
}

TEST_F(MnistTest, LocallyConnected1D)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(784), LocallyConnected(1, 7),
                                                FullyConnected(150, activation::sigmoid, L1Regularization(1e-5F)),
                                                FullyConnected(70), FullyConnected(10)});
    neuralNetwork.train(*data, 2_ep || 15_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.84F);
}

TEST_F(MnistTest, locallyConnected2D)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), LocallyConnected(2, 2), FullyConnected(150), FullyConnected(70), FullyConnected(10)});
    neuralNetwork.train(*data, 3_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70F);
}

TEST_F(MnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(4, 3, activation::ReLU), FullyConnected(10)});
    neuralNetwork.train(*data, 1_ep || 15_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.87F);
}

TEST_F(MnistTest, DISABLED_multipleFilterConvolutionBetterThanOnce)
{
    StraightforwardNeuralNetwork nn1Filter(
        {Input(1, 28, 28), Convolution(1, 5, activation::sigmoid), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.0002F, 0.8F));
    nn1Filter.train(*data, 1_ep || 10_s);
    auto accuracy1Filter = nn1Filter.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy1Filter, 0.6F);

    StraightforwardNeuralNetwork nn8Filters(
        {Input(1, 28, 28), Convolution(4, 5, activation::sigmoid), FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.0002F, 0.8F));
    nn8Filters.train(*data, 1_ep || 30_s);
    auto accuracy10Filters = nn8Filters.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy10Filters, 0.8F);

    EXPECT_GT(accuracy10Filters, accuracy1Filter);
}

TEST_F(MnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(12, 4, activation::ReLU), MaxPooling(2), FullyConnected(128, activation::ReLU),
         FullyConnected(10, activation::identity, Softmax())},
        StochasticGradientDescent(0.002F, 0.0F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForMNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*data, 0.9860_acc || 100_ep);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98F);
}

TEST_F(MnistTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForMNIST.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 261206);
    ASSERT_FLOAT_EQ(accuracy, 0.9871F);

    string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       ./resources/BestNeuralNetworkForMNIST.snn
 Parameters: 261206
 Epochs:     13
 Trainnig:   0
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 Convolution2D
                Input shape:  [1, 28, 28]
                Filters:      12
                Kernel size:  4x4
                Parameters:   204
                Activation:   ReLU
                Output shape: [12, 25, 25]
------------------------------------------------------------
 MaxPooling2D
                Input shape:  [12, 25, 25]
                Kernel size:  2x2
                Output shape: [12, 13, 13]
------------------------------------------------------------
 FullyConnected
                Input shape:  [2028]
                Neurons:      128
                Parameters:   259712
                Activation:   ReLU
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
                Learning rate: 0.002
                Momentum:      0
============================================================
)";
    string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}

TEST_F(MnistTest, DISABLED_SaveFeatureMap)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 28, 28), Convolution(9, 3, activation::sigmoid, ErrorMultiplier(100.0F)),
         LocallyConnected(4, 2, activation::sigmoid, ErrorMultiplier(100.0F)), FullyConnected(10)},
        StochasticGradientDescent(0.005F, 0.1F));
    neuralNetwork.saveData2DAsBitmap("./bitmap/data", *data, 13);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/before", *data, 13);
    neuralNetwork.train(*data, 1_ep);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90F);
    neuralNetwork.saveFilterLayersAsBitmap("./bitmap/after", *data, 13);
}