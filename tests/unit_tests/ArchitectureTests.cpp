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
        {Input(7, 8, 1), Convolution(2, 3, activation::ReLU), FullyConnected(25, activation::sigmoid)},
        {Input(10, 1), Convolution(1, 1), Convolution(2, 3)},
        {
            Input(10, 5, 3), Convolution(2, 2), Convolution(2, 2), FullyConnected(30, activation::gaussian), Convolution(2, 2),
            FullyConnected(15)
        },
        {Input(4, 20, 3), FullyConnected(30, activation::iSigmoid)},
        {Input(4, 2, 1, 2, 3), FullyConnected(5)}
    };

    for (auto&& Architecture : Architectures)
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
        {Input(), FullyConnected(10)},
        {FullyConnected(3, activation::sigmoid)},
        {Convolution(3, static_cast<int>(activation::sigmoid))},
        {Input(), FullyConnected(3, activation::sigmoid)},
        {Input(1, 0), FullyConnected(3, activation::sigmoid)},
        {Input(1, 1), FullyConnected(10), Input(6, 1)},
        {Input(10, 4, 1), Convolution(1, 7), FullyConnected(1)},
        {Input(8, 8, 8, 1), Convolution(1, 3), FullyConnected(2)}
    };

    vector<string> expectedErrorMessages =
    {
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: Neural Network must have at least 1 layer.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input LayerModel should be in first position.",
        "Invalid neural network architecture: Convolution matrix is too big.",
        "Invalid neural network architecture: Input with 3 dimensions or higher is not managed.",
    };

    for (int i = 0; i < Architectures.size(); i++)
    {
        try
        {
            StraightforwardNeuralNetwork neuralNetwork(Architectures[i]);
            FAIL();
        }
        catch (InvalidArchitectureException e)
        {
            ASSERT_EQ(e.what(), expectedErrorMessages[i]);
        }
    }
}

TEST(Architecture, NumberOfNeuronesAndParameters1)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(12, 12, 3),
            Convolution(3, 4),
            FullyConnected(50),
            Convolution(1, 3),
            FullyConnected(20),
            Recurrence(10, 4),
            FullyConnected(5)
        });
    const int numberOfNeurons = 9 * 9 * 3 + 50 + 48 + 20 + 10 + 5; // = 376
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);
    const int numberOfParameters = 9 * 9 * 3 * 16 * 3 + 50 * 9 * 9 * 3 + 48 * 3 + 20 * 48 + (20 + 1) * 10 + 10 * 5; // = 25178
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);
}

TEST(Architecture, NumberOfNeuronesAndParameters2)
{
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(12, 15, 3),
            LocallyConnected(3, 4),
            FullyConnected(200),
            LocallyConnected(1, 7),
            FullyConnected(2)
        });
    const int numberOfNeurons = 36 + 200 + 29 + 2; // = 297
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);
    const int numberOfParameters = 36 * 16 * 3 + 36 * 200 + 29 * 7 + 29 * 2; //  = 9189
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);
}

TEST(Architecture, InputWithSizeOf1)
{
    StraightforwardNeuralNetwork neuralNetworkFC({Input(5), FullyConnected(3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkLC({Input(5), LocallyConnected(2, 3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkC({Input(5), Convolution(2, 3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkR({Input(5), Recurrence(3, 1), FullyConnected(1)});

    ASSERT_EQ(neuralNetworkFC.isValid(), 0) << "FullyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkLC.isValid(), 0) << "LocallyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkC.isValid(), 0) << "Convolution neural network is invalid.";
    ASSERT_EQ(neuralNetworkR.isValid(), 0) << "Recurrence neural network is invalid.";
}

TEST(Architecture, InputWithSizeOf2)
{
    StraightforwardNeuralNetwork neuralNetworkFC({Input(5, 2), FullyConnected(3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkLC({Input(5, 2), LocallyConnected(2, 3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkC({Input(5, 2), Convolution(2,3), FullyConnected(1)});
    StraightforwardNeuralNetwork neuralNetworkR({Input(5, 2), Recurrence(3, 1), FullyConnected(1)});

    ASSERT_EQ(neuralNetworkFC.isValid(), 0) << "FullyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkLC.isValid(), 0) << "LocallyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkC.isValid(), 0) << "Convolution neural network is invalid.";
    ASSERT_EQ(neuralNetworkR.isValid(), 0) << "Recurrence neural network is invalid.";
}