#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "FashionMnist.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class FashionMnistTest : public testing::Test
{
protected :
    static void SetUpTestSuite()
    {
        FashionMnist dataset;
        data = move(dataset.data);
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> FashionMnistTest::data = nullptr;

TEST_F(FashionMnistTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 784);
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 60000);
    ASSERT_EQ(data->sets[training].labels.size(), 60000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
}

TEST_F(FashionMnistTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        784,
        {
            AllToAll(150),
            AllToAll(70),
            AllToAll(10)
        });
    neuralNetwork.trainingStart(*data);
    neuralNetwork.waitFor(1_ep || 180_s);
    neuralNetwork.trainingStop();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.70);
}