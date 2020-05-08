#include "../../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "AudioCatsAndDogs.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

class AudioCatsAndDogsTest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        AudioCatsAndDogs dataset("./datasets/AudioCatsAndDogs");
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
    ASSERT_EQ(data->sizeOfData, 16);
    ASSERT_EQ(data->numberOfLabel, 2);
    ASSERT_EQ(data->sets[training].inputs.size(), 210);
    ASSERT_EQ(data->sets[training].labels.size(), 210);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 67);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 67);
    ASSERT_TRUE(data->isValid());
}

TEST_F(AudioCatsAndDogsTest, trainNeuralNetwork)
{
    StraightforwardNeuralNetwork neuralNetwork({
        Input(16),
        Recurrent(30, 100),
        AllToAll(2)
    });
    neuralNetwork.startTraining(*data);
    neuralNetwork.waitFor(10_ep);
    neuralNetwork.stopTraining();
    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 0.5);
}