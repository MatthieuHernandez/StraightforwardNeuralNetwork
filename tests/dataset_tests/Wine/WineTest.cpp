#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Wine.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class WineTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        Wine dataset("./datasets/Wine");
        data = move(dataset.data);
    }

    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> WineTest::data = nullptr;

TEST_F(WineTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 13);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 178);
    ASSERT_EQ(data->sets[training].labels.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 178);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(13),
        AllToAll(20),
        AllToAll(8),
        AllToAll(3)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(1.00_acc || 3_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}