#include <snn/data/Data.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

TEST(Memory, passingArgByCopy)
{
    try
    {
        auto inputData = new vector2D<float>{{0, 0, 0}, {1, 1, 1}};
        auto expectedOutputs = new vector2D<float>{{0}, {1}};
        auto data = new Data(problem::regression, *inputData, *expectedOutputs);
        const vector<LayerModel> achitecture = {Input(1, 3),          Convolution(500, 1), FullyConnected(3000),
                                                FullyConnected(3000), Convolution(1, 4),   FullyConnected(1)};
        StraightforwardNeuralNetwork neuralNetwork(achitecture);

        delete inputData;
        delete expectedOutputs;

        neuralNetwork.startTrainingAsync(*data);
        neuralNetwork.waitFor(3_ms);
        neuralNetwork.stopTrainingAsync();
    }
    catch (exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}

TEST(Memory, copyOperator)
{
    try
    {
        vector2D<float> inputData = {{0, 0, 0}, {1, 1, 1}};
        vector2D<float> expectedOutputs = {{0}, {1}};
        Data data(problem::regression, inputData, expectedOutputs);
        auto neuralNetwork = new StraightforwardNeuralNetwork(
            {Input(1, 3), Convolution(500, 1), FullyConnected(250), FullyConnected(1)});

        StraightforwardNeuralNetwork neuralNetworkCopy = *neuralNetwork;
        delete neuralNetwork;

        neuralNetworkCopy.startTrainingAsync(data);
        neuralNetworkCopy.waitFor(3_ms);
        neuralNetworkCopy.stopTrainingAsync();
    }
    catch (exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}
