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

public :
    unique_ptr<Data> data;
};

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
    StraightforwardNeuralNetwork neuralNetwork({784, 150, 70, 10});
    neuralNetwork.trainingStart(*data);
    float accuracy = 0;
    for(int i = 0; i < 180 && accuracy < 0.91; i++)
    {
        accuracy = neuralNetwork.getGlobalClusteringRate();
        this_thread::sleep_for(1s);
    }
    neuralNetwork.trainingStop();
    ASSERT_ACCURACY(accuracy, 0.92);
}