#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Cifar10.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class Cifar10Test : public testing::Test
{
protected :
    Cifar10Test()
    {
        Cifar10 dataset;
        data = move(dataset.data);
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> Cifar10Test::data = nullptr;

TEST_F(Cifar10Test, DISABLED_loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 3072);
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 50000);
    ASSERT_EQ(data->sets[training].labels.size(), 50000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
}

TEST_F(Cifar10Test, DISABLED_trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        3072,
        { 
            AllToAll(200),
            AllToAll(80),
            AllToAll(10)
        });
    neuralNetwork.trainingStart(*data);
    neuralNetwork.waitFor(1_ep || 60_s);
    neuralNetwork.trainingStop();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.30);
}