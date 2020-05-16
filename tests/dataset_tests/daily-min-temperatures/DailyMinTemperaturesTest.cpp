#include "ExtendedGTest.hpp"
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
        DailyMinTemperatures dataset("./datasets/daily-min-temperatures");
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
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(10, 5),
        AllToAll(1, snn::identity)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(3_s || 0.8_mae);
    neuralNetwork.stopTraining();
    neuralNetwork.startTraining(*data);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 0.8);
}