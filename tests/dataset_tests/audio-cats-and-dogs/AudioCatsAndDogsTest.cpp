#include "ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "AudioCatsAndDogs.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

const static unsigned int sizeOfOneData = 1024;

class AudioCatsAndDogsTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        AudioCatsAndDogs dataset("./datasets/audio-cats-and-dogs", sizeOfOneData);
        data = move(dataset.data);
    }
    
    void SetUp() override
    {
        ASSERT_TRUE(data) << "Don't forget to download dataset";
    }
    
    static unique_ptr<Data> data;
};

unique_ptr<Data> AudioCatsAndDogsTest::data = nullptr;

TEST_F(AudioCatsAndDogsTest, loadData)
{
    ASSERT_EQ(data->sizeOfData, sizeOfOneData);
    ASSERT_EQ(data->numberOfLabel, 2);
    ASSERT_EQ(data->sets[training].numberOfTemporalSequence, 210);
    ASSERT_EQ(data->sets[snn::testing].numberOfTemporalSequence, 67);
    ASSERT_EQ(data->isValid(), 0);
}

TEST_F(AudioCatsAndDogsTest, trainNeuralNetwork)
{

    StraightforwardNeuralNetwork neuralNetwork({
        Input(sizeOfOneData),
        LocallyConnected(1, 16),
        Recurrence(300, 100),
        FullyConnected(30),
        Recurrence(2, 2)
    });
    /*auto numberOfparameters = neuralNetwork.getNumberOfParameters();
    PRINT_LOG("The number of parameter is " + to_string(numberOfparameters) + ".");*/
    neuralNetwork.optimizer.learningRate = 0.05f;
    neuralNetwork.optimizer.momentum = 0.0f;
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(5_ep);
    neuralNetwork.stopTraining();
    auto recall = neuralNetwork.getWeightedClusteringRate();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_RECALL(recall, 0.55);
    ASSERT_ACCURACY(accuracy, 0.60);
}