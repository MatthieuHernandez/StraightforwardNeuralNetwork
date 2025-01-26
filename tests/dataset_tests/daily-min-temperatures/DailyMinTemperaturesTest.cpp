#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "DailyMinTemperatures.hpp"
#include "ExtendedGTest.hpp"

using namespace snn;

class DailyMinTemperaturesTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            DailyMinTemperatures dataset("./resources/datasets/daily-min-temperatures", 5);
            data = move(dataset.data);
        }

        void SetUp() final { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static std::unique_ptr<Data> data;
};

std::unique_ptr<Data> DailyMinTemperaturesTest::data = nullptr;

TEST_F(DailyMinTemperaturesTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 1);
    ASSERT_EQ(data->numberOfLabels, 1);
    ASSERT_EQ((int)data->set.training.inputs.size(), 3649);
    ASSERT_EQ((int)data->set.training.labels.size(), 3649);
    ASSERT_EQ((int)data->set.testing.inputs.size(), 3649);
    ASSERT_EQ((int)data->set.testing.labels.size(), 3649);
    ASSERT_EQ(data->set.testing.numberOfTemporalSequence, 1);
    ASSERT_EQ(data->set.testing.numberOfTemporalSequence, 1);
    ASSERT_EQ(data->numberOfLabels, 1);
    ASSERT_EQ(data->isValid(), errorType::noError);
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(20), FullyConnected(1, activation::identity)},
                                               StochasticGradientDescent(0.004F, 0.2F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*data, 4_s || 2.0_mae, 1, 5);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 2.0);
}