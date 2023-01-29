#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "FashionMnist.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class FashionMnistTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        FashionMnist dataset("./datasets/Fashion-MNIST");
        data = move(dataset.data);
    }

    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> FashionMnistTest::data = nullptr;

TEST_F(FashionMnistTest, loadData)
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

TEST_F(FashionMnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    });
    neuralNetwork.train(*data, 1_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.76f);
}

TEST_F(FashionMnistTest, convolutionNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1, 28, 28),
        Convolution(1,3, activation::ReLU),
        FullyConnected(10)
        },
        StochasticGradientDescent(0.01f, 0.0f));
    neuralNetwork.train(*data, 1_ep || 20_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.74);
}

TEST_F(FashionMnistTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1, 28, 28),
        Convolution(8, 3),
        MaxPooling(2),
        FullyConnected(64),
        FullyConnected(10)
    },
        StochasticGradientDescent(0.005f, 0.0f));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForFashion-MNIST.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*data, 0.90_acc); // Reach after 109 epochs of 2 sec.

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.20f);
}

TEST_F(FashionMnistTest, DISABLED_valuateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./BestNeuralNetworkForFashion-MNIST.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 57390);
    ASSERT_FLOAT_EQ(accuracy, 0.8919f);
}
