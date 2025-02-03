#include <gtest/gtest.h>

#include <snn/data/Dataset.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

using namespace snn;

// NOLINTBEGIN(cppcoreguidelines-owning-memory)
TEST(Memory, passingArgByCopy)
{
    try
    {
        auto* inputData = new vector2D<float>{{0, 0, 0}, {1, 1, 1}};
        auto* expectedOutputs = new vector2D<float>{{0}, {1}};
        auto* dataset = new Dataset(problem::regression, *inputData, *expectedOutputs);
        const std::vector<LayerModel> achitecture = {Input(1, 3),          Convolution(500, 1), FullyConnected(3000),
                                                     FullyConnected(3000), Convolution(1, 4),   FullyConnected(1)};
        StraightforwardNeuralNetwork neuralNetwork(achitecture);

        delete inputData;
        delete expectedOutputs;

        neuralNetwork.startTrainingAsync(*dataset);
        neuralNetwork.waitFor(3_ms);
        neuralNetwork.stopTrainingAsync();
    }
    catch (std::exception& e)
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
        Dataset dataset(problem::regression, inputData, expectedOutputs);
        auto* neuralNetwork = new StraightforwardNeuralNetwork(
            {Input(1, 3), Convolution(500, 1), FullyConnected(250), FullyConnected(1)});

        StraightforwardNeuralNetwork neuralNetworkCopy = *neuralNetwork;
        delete neuralNetwork;

        neuralNetworkCopy.startTrainingAsync(dataset);
        neuralNetworkCopy.waitFor(3_ms);
        neuralNetworkCopy.stopTrainingAsync();
    }
    catch (std::exception& e)
    {
        EXPECT_TRUE(false) << e.what();
    }
    EXPECT_TRUE(true);
}
// NOLINTEND(cppcoreguidelines-owning-memory)
