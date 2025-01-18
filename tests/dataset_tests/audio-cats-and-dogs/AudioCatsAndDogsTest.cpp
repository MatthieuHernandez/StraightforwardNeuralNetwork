#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "AudioCatsAndDogs.hpp"
#include "ExtendedGTest.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

const static int sizeOfOneData = 16000;

class AudioCatsAndDogsTest : public testing::Test
{
    protected:
        static void SetUpTestSuite()
        {
            AudioCatsAndDogs dataset("./resources/datasets/audio-cats-and-dogs", sizeOfOneData);
            data = move(dataset.data);
        }

        void SetUp() override { ASSERT_TRUE(data) << "Don't forget to download dataset"; }

        static unique_ptr<Data> data;
};

unique_ptr<Data> AudioCatsAndDogsTest::data = nullptr;

TEST_F(AudioCatsAndDogsTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, sizeOfOneData);
    ASSERT_EQ(data->numberOfLabels, 2);
    ASSERT_EQ(data->sets[training].numberOfTemporalSequence, 210);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 67);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(AudioCatsAndDogsTest, DISABLED_trainBestNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(sizeOfOneData), MaxPooling(160), GruLayer(30), FullyConnected(2, activation::identity, Softmax())},
        StochasticGradientDescent(1e-6f, 0.99f));
    auto optimizer = std::dynamic_pointer_cast<internal::StochasticGradientDescent>(neuralNetwork.optimizer);
    neuralNetwork.autoSaveFilePath = "BestNeuralNetworkForAudioCatsAndDogs.snn";
    neuralNetwork.autoSaveWhenBetter = true;
    neuralNetwork.train(*data, 2000_ep, 1, 100);
    optimizer->learningRate *= 5;
    neuralNetwork.train(*data, 1.0_acc, 1, 100);

    auto recall = neuralNetwork.getWeightedClusteringRate();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_RECALL(recall, 0.51f);
    ASSERT_ACCURACY(accuracy, 0.6f);
}

TEST_F(AudioCatsAndDogsTest, evaluateBestNeuralNetwork)
{
    auto neuralNetwork = StraightforwardNeuralNetwork::loadFrom("./BestNeuralNetworkForAudioCatsAndDogs.snn");
    auto numberOfParameters = neuralNetwork.getNumberOfParameters();
    neuralNetwork.evaluate(*data);
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_EQ(numberOfParameters, 9242);
    ASSERT_FLOAT_EQ(accuracy, 0.91044772f);

    string expectedSummary =
        R"(============================================================
| SNN Model Summary                                        |
============================================================
 Name:       BestNeuralNetworkForAudioCatsAndDogs.snn
 Parameters: 9242
 Epochs:     6100
 Trainnig:   0
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
                Input shape:  [30]
                Neurons:      30
                Parameters:   9180
                Output shape: [100]
------------------------------------------------------------
 FullyConnected
                Input shape:  [30]
                Neurons:      2
                Parameters:   62
                Activation:   identity
                Output shape: [2]
                Optimizers:   Softmax
============================================================
|  Optimizer                                               |
============================================================
 StochasticGradientDescent
                Learning rate: 5e-06
                Momentum:      0.99
============================================================
)";
    string summary = neuralNetwork.summary();
    ASSERT_EQ(summary, expectedSummary);
}
