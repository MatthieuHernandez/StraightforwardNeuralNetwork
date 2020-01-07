
#include "GTestTools.hpp"
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
        auto option = new StraightforwardOption();
        StraightforwardNeuralNetwork neuralNetwork({3, 5, 1}, {sigmoid, sigmoid}, *option);

        delete inputData;
        delete expectedOutputs;
        delete option;

        neuralNetwork.trainingStart(*data);
        this_thread::sleep_for(1ms);
        neuralNetwork.trainingStop();
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
        auto neuralNetwork = new StraightforwardNeuralNetwork({3, 5, 1});
        
        StraightforwardNeuralNetwork neuralNetworkCopy = *neuralNetwork;
        delete neuralNetwork;

        neuralNetworkCopy.trainingStart(data);
        this_thread::sleep_for(1ms);
        neuralNetworkCopy.trainingStop();
    }
    catch(const std::exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}
