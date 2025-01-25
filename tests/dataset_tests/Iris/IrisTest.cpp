#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "Iris.hpp"

using namespace snn;

class IrisTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            Iris dataset("./resources/datasets/Iris");
            data = move(dataset.data);
        }

        void SetUp() final { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static std::unique_ptr<Data> data;
};

std::unique_ptr<Data> IrisTest::data = nullptr;

TEST_F(IrisTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, 4);
    ASSERT_EQ(data->numberOfLabels, 3);
    ASSERT_EQ((int)data->sets[training].inputs.size(), 150);
    ASSERT_EQ((int)data->sets[training].labels.size(), 150);
    ASSERT_EQ((int)data->sets[snn::testing].inputs.size(), 150);
    ASSERT_EQ((int)data->sets[snn::testing].labels.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 0);
    ASSERT_EQ(data->isValid(), ErrorType::noError);
}

TEST_F(IrisTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(4), FullyConnected(15), FullyConnected(5), FullyConnected(3)});

    PRINT_NUMBER_OF_PARAMETERS(neuralNetwork.getNumberOfParameters());

    neuralNetwork.train(*data, 0.98_acc || 2_s);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.98F);
}
