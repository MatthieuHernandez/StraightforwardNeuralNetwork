#include <thread>
#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Iris.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class IrisTest : public testing::Test
{
protected:
    IrisTest()
    {
        Iris dataset;
        data = move(dataset.data);
    }

public:
    shared_ptr<Data> data;
};

TEST_F(IrisTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 4);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 150);
    ASSERT_EQ(data->sets[training].labels.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 150);
}

TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        4,
        {
            AllToAll(15),
            AllToAll(5),
            AllToAll(3)
        });
    neuralNetwork.trainingStart(*data);
    this_thread::sleep_for(2s);
    neuralNetwork.trainingStop();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98);
}