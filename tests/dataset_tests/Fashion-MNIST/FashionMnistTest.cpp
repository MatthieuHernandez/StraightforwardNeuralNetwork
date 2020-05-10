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
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 60000);
    ASSERT_EQ(data->sets[training].labels.size(), 60000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
    ASSERT_TRUE(data->isValid());
}

TEST_F(FashionMnistTest, feedforwardNeuralNetwork)
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
    ASSERT_ACCURACY(accuracy, 0.76);
}

TEST_F(FashionMnistTest, convolutionNeuralNetwork)
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
    ASSERT_ACCURACY(accuracy, 0.75);
}