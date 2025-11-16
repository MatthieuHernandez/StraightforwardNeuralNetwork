#include <gtest/gtest.h>

#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "ExtendedGTest.hpp"
#include "KeywordSpotting.hpp"

using namespace snn;

const static int sizeOfOneData = 1600;

class KeywordSpottingTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            KeywordSpotting datasetTest("./resources/datasets/KWS", sizeOfOneData);
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> KeywordSpottingTest::dataset = nullptr;

TEST_F(KeywordSpottingTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, sizeOfOneData);
    ASSERT_EQ(dataset->numberOfLabels, 3);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 1649);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 413);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(KeywordSpottingTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({Input(sizeOfOneData), Recurrence(200, snn::activation::ReLU),
                                                FullyConnected(3, snn::activation::identity, Softmax())},
                                               StochasticGradientDescent(1e-3F, 0.8F));
    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForKWS.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 0.9_acc);

    auto recall = neuralNetwork.getWeightedClusteringRate();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_RECALL(recall, 0.51F);
    ASSERT_ACCURACY(accuracy, 0.6F);
}
