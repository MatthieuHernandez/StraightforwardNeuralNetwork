#include "../ExtendedGTest.hpp"
#include "tools/ExtendedExpection.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

TEST(Architecture, ValidArchitectures)
{
    vector2D<LayerModel> Architectures =
    {
        {Input(7, 8, 1), Convolution(2, 3, ReLU), AllToAll(25, sigmoid)},
        //{Input(10, 1), Convolution(1, 1), Convolution(2, 3)},
        {Input(10, 5, 3), Convolution(2, 2), Convolution(2, 2), AllToAll(30, gaussian), Convolution(2, 2), AllToAll(15)},
        {Input(4, 20, 3), AllToAll(30, iSigmoid)},
        {Input(8, 8, 8, 1), AllToAll(5)}
    };

    for(auto&& Architecture : Architectures)
    {
        StraightforwardNeuralNetwork neuralNetwork(Architecture);
        ASSERT_EQ(neuralNetwork.isValid(), 0);
    }
}

TEST(Architecture, invalidArchitectures)
{
    vector2D<LayerModel> Architectures =
    {
        {},
        {Input(7, 7, 1)},
        {Input(), AllToAll(10)},
        {AllToAll(3, sigmoid)},
        {Convolution(3, sigmoid)},
        {Input(), AllToAll(3, sigmoid)},
        {Input(1, 0), AllToAll(3, sigmoid)},
        {Input(1, 1), AllToAll(10), Input(6, 1)},
        {Input(10, 4, 1), Convolution(1, 7), AllToAll(1)},
        {Input(8, 8, 8, 1), Convolution(1, 3), AllToAll(2)}
    };

    vector<string> expectedErrorMessages =
    {
        "Invalid neural network architecture: Neural Network must have at least 1 layer.",
        "Invalid neural network architecture: Neural Network must have at least 1 layer.",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input LayerModel should be in first position.",
        "Invalid neural network architecture: Convolution matrix is too big.",
        "Invalid neural network architecture: Layer type is not implemented.",
    };

    for(int i = 0; i < Architectures.size(); i++)
    {
        try
        {
            StraightforwardNeuralNetwork neuralNetwork(Architectures[i]);
            FAIL();
        }
        catch(InvalidAchitectureException e)
        {
            ASSERT_EQ(e.what(), expectedErrorMessages[i]);
        }
    }
}