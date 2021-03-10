#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Mnist.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class MnistTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        Mnist dataset("./datasets/MNIST");
        data = move(dataset.data);
    }

    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }

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
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(MnistTest, simplierNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        MaxPooling(2),
        FullyConnected(10)
    },
        StochasticGradientDescent(0.02f, 0.1f));
    neuralNetwork.train(*data, 1_ep || 3_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.86f);
}

TEST_F(MnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(150, activation::sigmoid, Dropout(0.1f)),
        FullyConnected(70),
        FullyConnected(10)
    },
        StochasticGradientDescent(0.01f, 0.9f));
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 35_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.91f);
}


TEST_F(MnistTest, feedforwardNeuralNetworkWithAdam)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    },
        Adam());
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(10_ep /*|| 55_s*/);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.91f);
}

TEST_F(MnistTest, feedforwardNeuralNetworkWithGRU)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(100, activation::sigmoid, L2Regularization(1e-4f)),
        GruLayer(10),
        FullyConnected(10)
    });
    neuralNetwork.train(*data, 1_ep || 35_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90f);
}

TEST_F(MnistTest, LocallyConnected1D)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        LocallyConnected(1, 7),
        FullyConnected(150, activation::sigmoid, L1Regularization(1e-5f)),
        FullyConnected(70),
        FullyConnected(10)
    });
    neuralNetwork.train(*data, 2_ep || 35_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.84f);
}

TEST_F(MnistTest, LocallyConnected2D)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        LocallyConnected(2, 2),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    });
    neuralNetwork.train(*data, 3_ep || 45_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70f);
}

TEST_F(MnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(1,5),
        FullyConnected(70),
        FullyConnected(10)
        });
    neuralNetwork.train(*data, 1_ep || 35_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.93f);
}

TEST_F(MnistTest, multipleLayersNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        LocallyConnected(1, 2, activation::sigmoid),
        Convolution(1, 2, activation::sigmoid),
        FullyConnected(70),
        Convolution(1, 4, activation::sigmoid),
        FullyConnected(10)
    },
        StochasticGradientDescent(0.03f, 0.90f));

    neuralNetwork.train(*data, 2_ep || 60_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70f);
}

TEST_F(MnistTest, multipleFilterConvolutionBetterThanOnce)
{
    StraightforwardNeuralNetwork nn1Filter({
        Input(28, 28, 1),
        Convolution(1,26, activation::sigmoid),
        FullyConnected(10)
        });
    nn1Filter.train(*data, 1_ep || 60_s);
    auto accuracy1Filter = nn1Filter.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy1Filter, 0.7f);

    StraightforwardNeuralNetwork nn10Filters({
        Input(28, 28, 1),
        Convolution(8,26, activation::sigmoid),
        FullyConnected(10)
        });
    nn10Filters.train(*data, 1_ep || 60_s);
    auto accuracy10Filters = nn10Filters.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy10Filters, 0.8f);

    EXPECT_GT(accuracy10Filters, accuracy1Filter);
}

TEST_F(MnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(2,4),
        FullyConnected(150),
        FullyConnected(10)
        },
        StochasticGradientDescent(0.02f, 0.6f));  

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForMNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*data, 0.9859_acc);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20f);
}

TEST_F(MnistTest, EvaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./BestNeuralNetworkForMNIST.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 209000);
    ASSERT_FLOAT_EQ(accuracy, 0.9862f);
}