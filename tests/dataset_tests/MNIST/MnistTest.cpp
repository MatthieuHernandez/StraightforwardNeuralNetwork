#include "../../ExtendedGTest.hpp"
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
    ASSERT_TRUE(data->isValid());
}

TEST_F(MnistTest, feedforwardNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(784),
        AllToAll(150),
        AllToAll(70),
        AllToAll(10)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.90);
}

TEST_F(MnistTest, convolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(1,5),
        AllToAll(70),
        AllToAll(10)
        });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1_ep || 45_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.93);
}

TEST_F(MnistTest, multipleConvolutionalNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(28, 28, 1),
        Convolution(1,4),
        Convolution(1,4),
        AllToAll(70),
        Convolution(1, 4, sigmoid),
        AllToAll(10)
        });
    neuralNetwork.startTraining(*data);
    #ifdef DEBUG
        neuralNetwork.waitFor(1_ep);
    #else
        neuralNetwork.waitFor(1_ep || 60_s);
    #endif
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.80);
}