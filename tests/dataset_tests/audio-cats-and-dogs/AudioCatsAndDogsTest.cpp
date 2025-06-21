#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "AudioCatsAndDogs.hpp"
#include "ExtendedGTest.hpp"

using namespace snn;

const static int sizeOfOneData = 16000;

class AudioCatsAndDogsTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            AudioCatsAndDogs datasetTest("./resources/datasets/audio-cats-and-dogs", sizeOfOneData);
            dataset = std::move(datasetTest.dataset);
        }

        void SetUp() final { ASSERT_TRUE(dataset) << "Don't forget to download dataset"; }

        static std::unique_ptr<Dataset> dataset;
};

std::unique_ptr<Dataset> AudioCatsAndDogsTest::dataset = nullptr;

TEST_F(AudioCatsAndDogsTest, loadData)
{
    ASSERT_EQ(dataset->sizeOfData, sizeOfOneData);
    ASSERT_EQ(dataset->numberOfLabels, 2);
    ASSERT_EQ(dataset->data.training.numberOfTemporalSequence, 210);
    ASSERT_EQ(dataset->data.testing.numberOfTemporalSequence, 67);
    ASSERT_EQ(dataset->isValid(), errorType::noError);
}

TEST_F(AudioCatsAndDogsTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(sizeOfOneData), MaxPooling(160), GruLayer(10), FullyConnected(2, snn::activation::sigmoid)},
        StochasticGradientDescent(1e-5F, 0.99F));
    auto optimizer = std::dynamic_pointer_cast<internal::StochasticGradientDescent>(neuralNetwork.optimizer);
    neuralNetwork.autoSaveFilePath = "./resources/BestNeuralNetworkForAudioCatsAndDogs.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*dataset, 1.0_acc, 1, 200);  // Achieved after 2400 epochs, ~0.1 second each.

    auto recall = neuralNetwork.getWeightedClusteringRate();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_RECALL(recall, 0.51F);
    ASSERT_ACCURACY(accuracy, 0.6F);
}

TEST_F(AudioCatsAndDogsTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./resources/BestNeuralNetworkForAudioCatsAndDogs.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*dataset);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 3082);
    ASSERT_FLOAT_EQ(accuracy, 0.91044778F);

    const std::string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       ./resources/BestNeuralNetworkForAudioCatsAndDogs.snn
 Parameters: 3082
 Epochs:     4600
 Trainnig:   7341600
============================================================
| Layers                                                   |
============================================================
------------------------------------------------------------
 MaxPooling1D
                Input shape:  [1, 16000]
                Kernel size:  160
                Output shape: [1, 100]
------------------------------------------------------------
 GruLayer
                Input shape:  [10]
                Neurons:      10
                Parameters:   3060
                Output shape: [100]
------------------------------------------------------------
 FullyConnected
                Input shape:  [10]
                Neurons:      2
                Parameters:   22
                Activation:   sigmoid
                Output shape: [2]
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 1e-05
                Momentum:      0.99
============================================================
)";
    const std::string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}
