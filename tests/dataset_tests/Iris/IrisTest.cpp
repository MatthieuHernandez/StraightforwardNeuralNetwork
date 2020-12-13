#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "Iris.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class IrisTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        Iris dataset("./datasets/Iris");
        data = move(dataset.data);
    }

    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }

    static unique_ptr<Data> data;
};

unique_ptr<Data> IrisTest::data = nullptr;

TEST_F(IrisTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 4);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 150);
    ASSERT_EQ((int)data->sets[training].labels.size(), 150);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 150);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(15),
        FullyConnected(5),
        FullyConnected(3)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(0.98_acc || 2_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRateMax();
    ASSERT_ACCURACY(accuracy, 0.98f);
}
