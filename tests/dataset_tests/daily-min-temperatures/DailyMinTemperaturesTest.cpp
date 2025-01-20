#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "DailyMinTemperatures.hpp"
#include "ExtendedGTest.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class DailyMinTemperaturesTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            DailyMinTemperatures dataset("./resources/datasets/daily-min-temperatures", 5);
            data = move(dataset.data);
        }

        void SetUp() override { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static unique_ptr<Data> data;
};

unique_ptr<Data> DailyMinTemperaturesTest::data = nullptr;

TEST_F(DailyMinTemperaturesTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 1);
    ASSERT_EQ(data->numberOfLabels, 1);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 3649);
    ASSERT_EQ((int)data->sets[training].labels.size(), 3649);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 3649);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 3649);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 1);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 1);
    ASSERT_EQ(data->numberOfLabels, 1);
    ASSERT_EQ(data->isValid(), ErrorType::noError);
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(20), FullyConnected(1, activation::identity)},
                                               StochasticGradientDescent(0.004f, 0.2f));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*data, 4_s || 2.0_mae, 1, 5);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 2.0);
}