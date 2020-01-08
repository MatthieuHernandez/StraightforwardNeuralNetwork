#include <thread>
#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Mnist.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class MnistTest : public testing::Test
{
protected :
    MnistTest()
    {
        Mnist dataset;
        data = move(dataset.data);
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> MnistTest::data = nullptr;

TEST_F(MnistTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 784);
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 60000);
    ASSERT_EQ(data->sets[training].labels.size(), 60000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
}

TEST_F(MnistTest, DISABLED_trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        784,
        {
            AllToAll(150),
            AllToAll(70),
            AllToAll(10)
        });
    neuralNetwork.trainingStart(*data);
    neuralNetwork.waitFor(1_ep || 30_s);
    neuralNetwork.trainingStop();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.92);
}