#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Wine.hpp"

using namespace snn;

class WineTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Wine datasetTest("./resources/datasets/Wine");
            dataset = move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> WineTest::dataset = nullptr;

TEST_F(WineTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 13);
    ASSERT_EQ(dataset->numberOfLabels, 3);
    ASSERT_EQ(dataset->data.training.inputs.size(), 178);
    ASSERT_EQ(dataset->data.training.labels.size(), 178);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 178);
    ASSERT_EQ(dataset->data.testing.labels.size(), 178);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(13), FullyConnected(20), FullyConnected(8), FullyConnected(3)});

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*dataset, 1.00_acc || 2_s, 1, 4);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}