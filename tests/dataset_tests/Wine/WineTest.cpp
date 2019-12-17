#include <thread>
#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Wine.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class WineTest : public testing::Test
{
protected :
    WineTest()
    {
        Wine dataset;
        data = move(dataset.data);
    }

public :
    shared_ptr<Data> data;
};

TEST_F(WineTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 13);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 178);
    ASSERT_EQ(data->sets[training].labels.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 178);
}

TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({13, 20, 8, 3});
    neuralNetwork.trainingStart(*data);
    this_thread::sleep_for(3s);
    neuralNetwork.trainingStop();
    const auto accuracy = neuralNetwork.getGlobalClusteringRate();
    //PRINT_LOG("accuracy = " + to_string(accuracy));
    ASSERT_TRUE(accuracy > 0.98);
}