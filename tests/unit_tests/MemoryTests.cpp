
#include "../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/DataForRegression.hpp"

using namespace std;
using namespace snn;

TEST(Memory, passingArgByCopy)
{
    try
    {
        auto inputData = new vector<vector<float>> {{0, 0, 0}, {1, 1, 1}};
        auto expectedOutputs = new vector<vector<float>> {{0}, {1}};
        auto data = new DataForRegression(*inputData, *expectedOutputs, 0.1);
        vector<LayerModel> model = {
            AllToAll(4000), 
            AllToAll(4000),
            AllToAll(4000),
            AllToAll(1)
        };
        StraightforwardNeuralNetwork neuralNetwork(3, model);

        delete inputData;
        delete expectedOutputs;

        neuralNetwork.startTraining(*data);
        this_thread::sleep_for(1ms);
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
        DataForRegression data(inputData, expectedOutputs, 0.1);
        auto neuralNetwork = new StraightforwardNeuralNetwork(3, {AllToAll(500), AllToAll(250)});
        
        StraightforwardNeuralNetwork neuralNetworkCopy = *neuralNetwork;
        delete neuralNetwork;

        neuralNetworkCopy.startTraining(data);
        this_thread::sleep_for(1ms);
        neuralNetworkCopy.stopTraining();
    }
    catch(const std::exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}
