#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Wine.hpp"

using namespace snn;

class WineTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Wine dataset("./resources/datasets/Wine");
            data = move(dataset.data);
        }

        void SetUp() final { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static std::unique_ptr<Data> data;
};

std::unique_ptr<Data> WineTest::data = nullptr;

TEST_F(WineTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 13);
    ASSERT_EQ(data->numberOfLabels, 3);
    ASSERT_EQ((int)data->set.training.inputs.size(), 178);
    ASSERT_EQ((int)data->set.training.labels.size(), 178);
    ASSERT_EQ((int)data->set.testing.inputs.size(), 178);
    ASSERT_EQ((int)data->set.testing.labels.size(), 178);
    ASSERT_EQ(data->set.training.numberOfTemporalSequence, 0);
    ASSERT_EQ(data->set.testing.numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), errorType::noError);
}

TEST_F(WineTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(13), FullyConnected(20), FullyConnected(8), FullyConnected(3)});

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*data, 1.00_acc || 2_s, 1, 4);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}