#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Iris.hpp"

using namespace snn;

class IrisTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Iris datasetTest("./resources/datasets/Iris");
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> IrisTest::dataset = nullptr;

TEST_F(IrisTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, 4);
    ASSERT_EQ(dataset->numberOfLabels, 3);
    ASSERT_EQ(dataset->data.training.inputs.size(), 150);
    ASSERT_EQ(dataset->data.training.labels.size(), 150);
    ASSERT_EQ(dataset->data.testing.inputs.size(), 150);
    ASSERT_EQ(dataset->data.testing.labels.size(), 150);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(15), FullyConnected(5), FullyConnected(3)});

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*dataset, 0.98_acc || 2_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98F);
}
