#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

Data createDataForConvolutionTests();

TEST(Convolution, SimpleConvolution2D)
{
    auto data = createDataForConvolutionTests();
    vector<LayerModel> architectures =
    {
        Input(3, 3, 2), Convolution(2, 2, activation::sigmoid),
        FullyConnected(2)
    };
    StraightforwardNeuralNetwork neuralNetwork(architectures, StochasticGradientDescent(0.5f));
    neuralNetwork.startTrainingAsync(data, 200_ep);
    float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);

}

Data createDataForConvolutionTests()
{
    vector<vector<float>> inputData = {
        {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 
         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
         -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},

        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},

        {0.11f, 0.12f, 0.21f, 0.22f, 0.31f, 0.32f, 
         0.41f, 0.42f, 0.51f, 0.52f, 0.61f, 0.62f,
         0.71f, 0.72f, 0.81f, 0.82f, 0.91f, 0.92f}
    };
    vector2D<float> expectedOutputs = {{0, 1}, {0, 1}, {1, 0}};

    return Data(problem::classification, inputData, expectedOutputs);
}