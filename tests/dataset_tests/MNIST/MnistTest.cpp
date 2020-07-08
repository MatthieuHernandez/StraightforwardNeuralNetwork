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
        Mnist dataset("./datasets/MNIST");;
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
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 60000);
    ASSERT_EQ(data->sets[training].labels.size(), 60000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(MnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90f);
}

TEST_F(MnistTest, LocallyConnected1D)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        LocallyConnected(1, 7),
        FullyConnected(150),
        FullyConnected(70),
        FullyConnected(10)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(2_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70f);
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
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(3_ep || 45_s);
    neuralNetwork.stopTraining();
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
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.93f);
}

TEST_F(MnistTest, multipleLayersNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        LocallyConnected(1, 2, sigmoid),
        Convolution(1, 2, sigmoid),
        FullyConnected(70),
        Convolution(1, 4, sigmoid),
        FullyConnected(10)
    });
    neuralNetwork.optimizer.learningRate = 0.06;
    neuralNetwork.optimizer.momentum = 0.9;
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(2_ep || 60_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70f);
}

TEST_F(MnistTest, multipleFilterConvolutionBetterThanOnce)
{
    StraightforwardNeuralNetwork nn1Filer({
        Input(28, 28, 1),
        Convolution(1,26, sigmoid),
        FullyConnected(10)
        });
    nn1Filer.startTraining(*data);
    nn1Filer.waitFor(1_ep );
    nn1Filer.stopTraining();
    auto accuracy1Filter = nn1Filer.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy1Filter, 0.7f);

    StraightforwardNeuralNetwork nn10Filers({
        Input(28, 28, 1),
        Convolution(8,26, sigmoid),
        FullyConnected(10)
        });
    nn10Filers.startTraining(*data);
    nn10Filers.waitFor(1_ep);
    nn10Filers.stopTraining();
    auto accuracy10Filers = nn10Filers.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy10Filers, 0.8f);

    EXPECT_GT(accuracy10Filers, accuracy1Filter);
}