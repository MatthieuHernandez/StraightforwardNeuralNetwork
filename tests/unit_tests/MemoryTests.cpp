#include "../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace snn;

TEST(Memory, passingArgByCopy)
{
    try
    {
        auto inputData = new vector<vector<float>> {{0, 0, 0}, {1, 1, 1}};
        auto expectedOutputs = new vector<vector<float>> {{0}, {1}};
        auto data = new Data(regression, *inputData, *expectedOutputs);
        vector<LayerModel> achitecture = {
            Input(3, 1),
            Convolution(500, 1),
            FullyConnected(3000),
            FullyConnected(3000),
            Convolution(1, 4),
            FullyConnected(1)
        };
        StraightforwardNeuralNetwork neuralNetwork(achitecture);

        delete inputData;
        delete expectedOutputs;

        neuralNetwork.startTraining(*data);
        neuralNetwork.waitFor(3_ms);
        neuralNetwork.stopTraining();
    }
    catch(const std::exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}

TEST(Memory, copyOperator)
{
    try
    {
        vector<vector<float>> inputData = {{0, 0, 0}, {1, 1, 1}};
        vector<vector<float>> expectedOutputs = {{0}, {1}};
        Data data(regression, inputData, expectedOutputs);
        auto neuralNetwork = new StraightforwardNeuralNetwork({
                Input(3, 1),
                Convolution(500, 1),
                FullyConnected(250),
                FullyConnected(1)
            });
        
        StraightforwardNeuralNetwork neuralNetworkCopy = *neuralNetwork;
        delete neuralNetwork;

        neuralNetworkCopy.startTraining(data);
        neuralNetworkCopy.waitFor(3_ms);
        neuralNetworkCopy.stopTraining();
    }
    catch(const std::exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}
