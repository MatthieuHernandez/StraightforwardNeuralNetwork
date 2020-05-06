#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "DailyMinTemperatures.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class DailyMinTemperaturesTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        DailyMinTemperature dataset("./datasets/DailyMinTemperature");
        data = move(dataset.data);
    }
    
    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }
    
    static unique_ptr<Data> data;
};

unique_ptr<Data> DailyMinTemperaturesTest::data = nullptr;

TEST_F(DailyMinTemperaturesTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 1);
    ASSERT_EQ(data->numberOfLabel, 1);
    ASSERT_EQ(data->sets[training].inputs.size(), 3649);
    ASSERT_EQ(data->sets[training].labels.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 3649);
    ASSERT_TRUE(data->isValid());
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrent(10, 5),
        AllToAll(1) // Adjusted Sigmoid or Linear function
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(0.8_mae || 3_s);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}