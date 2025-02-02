#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "DailyMinTemperatures.hpp"
#include "ExtendedGTest.hpp"

using namespace snn;

class DailyMinTemperaturesTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            DailyMinTemperatures datasetTest("./resources/datasets/daily-min-temperatures", 5);
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> DailyMinTemperaturesTest::dataset = nullptr;

TEST_F(DailyMinTemperaturesTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 1);
    ASSERT_EQ(dataset->numberOfLabels, 1);
    ASSERT_EQ(dataset->data.training.inputs.size(), 3649);
    ASSERT_EQ(dataset->data.training.labels.size(), 3649);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 3649);
    ASSERT_EQ(dataset->data.testing.labels.size(), 3649);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 1);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 1);
    ASSERT_EQ(dataset->numberOfLabels, 1);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(DailyMinTemperaturesTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(20), FullyConnected(1, activation::identity)},
                                               StochasticGradientDescent(0.004F, 0.2F));

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*dataset, 4_s || 2.0_mae, 1, 5);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    ASSERT_MAE(mae, 2.0);
}