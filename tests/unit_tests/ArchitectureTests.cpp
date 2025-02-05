#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

TEST(Architecture, ValidArchitectures)
{
    const vector2D<LayerModel> architectures = {
        {Input(1, 7, 8), Convolution(2, 3, activation::ReLU), FullyConnected(25, activation::sigmoid)},
        {Input(1, 10), Convolution(1, 1), Convolution(2, 3)},
        {Input(3, 10, 5), Convolution(2, 2), Convolution(2, 2), FullyConnected(30, activation::gaussian),
         Convolution(2, 2), FullyConnected(15)},
        {Input(3, 4, 20), FullyConnected(30, activation::iSigmoid)},
        {Input(2, 4, 2, 1, 2), FullyConnected(5)}};

    for (auto&& architecture : architectures)
    {
        const StraightforwardNeuralNetwork neuralNetwork(architecture);
        ASSERT_EQ(neuralNetwork.isValid(), errorType::noError);
    }
}

TEST(Architecture, InvalidArchitectures)
{
    vector2D<LayerModel> Architectures = {{},
                                          {Input(7, 7, 1)},
                                          {Input(), FullyConnected(10)},
                                          {FullyConnected(3, activation::sigmoid)},
                                          {Convolution(3, static_cast<int>(activation::sigmoid))},
                                          {Input(), FullyConnected(3, activation::sigmoid)},
                                          {Input(1, 0), FullyConnected(3, activation::sigmoid)},
                                          {Input(1, 1), FullyConnected(10), Input(6, 1)},
                                          {Input(4, 1, 10), Convolution(1, 7), FullyConnected(1)},
                                          {Input(8, 8, 8, 8), Convolution(1, 3), FullyConnected(2)}};

    const std::vector<std::string> expectedErrorMessages = {
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: Neural Network must have at least 1 layer.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: First LayerModel must be a Input type LayerModel.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input of layer has size of 0.",
        "Invalid neural network architecture: Input LayerModel should be in first position.",
        "Invalid neural network architecture: Convolution kernel is too big.",
        "Invalid neural network architecture: Input with 3 dimensions or higher is not managed.",
    };

    for (size_t i = 0; i < Architectures.size(); i++)
    {
        try
        {
            const StraightforwardNeuralNetwork neuralNetwork(Architectures[i]);
            FAIL();
        }
        catch (InvalidArchitectureException& e)
        {
            ASSERT_EQ(e.what(), expectedErrorMessages[i]);
        }
    }
}

TEST(Architecture, NumberOfNeuronesAndParameters1)
{
    const StraightforwardNeuralNetwork neuralNetwork({Input(3, 12, 12), MaxPooling(2), Convolution(3, 4),
                                                      FullyConnected(60), MaxPooling(3), Convolution(1, 3),
                                                      FullyConnected(20), Recurrence(10), FullyConnected(5)});
    constexpr int numberOfNeurons = 3 + 60 + 1 + 20 + 10 + 5;
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);        // 99
    constexpr int numberOfParameters = 0 +                                 // MaxPooling
                                       3 * ((4 * 4 * 3) + 1)               // Convolution
                                       + 60 * (3 * 9 + 1)                  // FullyConnected
                                       + 0                                 // MaxPooling
                                       + 1 * (3 + 1)                       // Convolution
                                       + 20 * (18 + 1)                     // FullyConnected
                                       + 10 * (20 + 2)                     // Recurrence
                                       + 5 * (10 + 1);                     // FullyConnected
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);  // 2390
}

TEST(Architecture, NumberOfNeuronesAndParameters2)
{
    const StraightforwardNeuralNetwork neuralNetwork(
        {Input(3, 12, 15), LocallyConnected(3, 4), FullyConnected(200), LocallyConnected(1, 7), FullyConnected(2)});
    const int numberOfNeurons = 36 + 200 + 29 + 2;  // = 267
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);
    const int numberOfParameters = 36 * 16 * 3 + 36 * 200 + 29 * 7 + 29 * 2 + 267;  // = 9456
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);
}

TEST(Architecture, NumberOfNeuronesAndParameters3)
{
    const StraightforwardNeuralNetwork neuralNetwork({Input(10), GruLayer(10), Recurrence(10)});
    const int numberOfNeurons = 10 + 10;  // = 20
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);
    const int numberOfParameters = (10 + 2) * 10 * 3 + (10 + 2) * 10;  // = 480
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);
}

TEST(Architecture, NumberOfNeuronesAndParameters4)
{
    const StraightforwardNeuralNetwork neuralNetwork(
        {Input(3, 15, 14), Convolution(16, 4, activation::ReLU), FullyConnected(9)});
    const int numberOfNeurons = 16 + 9;  // = 25
    ASSERT_EQ(neuralNetwork.getNumberOfNeurons(), numberOfNeurons);
    const int numberOfParameters = (16 * (4 * 4 * 3 + 1)) + ((16 * (12 * 11) + 1) * 9);  // = 937
    ASSERT_EQ(neuralNetwork.getNumberOfParameters(), numberOfParameters);
}

TEST(Architecture, InputWithSizeOf1)
{
    const StraightforwardNeuralNetwork neuralNetworkFC({Input(5), FullyConnected(3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkLC({Input(5), LocallyConnected(2, 3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkC({Input(5), Convolution(2, 3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkR({Input(5), Recurrence(3), FullyConnected(1)});

    ASSERT_EQ(neuralNetworkFC.isValid(), errorType::noError) << "FullyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkLC.isValid(), errorType::noError) << "LocallyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkC.isValid(), errorType::noError) << "Convolution neural network is invalid.";
    ASSERT_EQ(neuralNetworkR.isValid(), errorType::noError) << "Recurrence neural network is invalid.";
}

TEST(Architecture, InputWithSizeOf2)
{
    const StraightforwardNeuralNetwork neuralNetworkFC({Input(2, 5), FullyConnected(3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkLC({Input(2, 5), LocallyConnected(2, 3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkC({Input(2, 5), Convolution(2, 3), FullyConnected(1)});
    const StraightforwardNeuralNetwork neuralNetworkR({Input(2, 5), Recurrence(3), FullyConnected(1)});

    ASSERT_EQ(neuralNetworkFC.isValid(), errorType::noError) << "FullyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkLC.isValid(), errorType::noError) << "LocallyConnected neural network is invalid.";
    ASSERT_EQ(neuralNetworkC.isValid(), errorType::noError) << "Convolution neural network is invalid.";
    ASSERT_EQ(neuralNetworkR.isValid(), errorType::noError) << "Recurrence neural network is invalid.";
}
