#include <memory>
#include <numeric>
#include <boost/serialization/smart_cast.hpp>

#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

Data createDataForConvolutionTests();

TEST(Convolution, LayerConvolution2D)
{
    vector<float> input(50);
    vector<float> kernel0 {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
    vector<float> kernel1 {10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18};
    vector<float> error(9);
    std::iota(std::begin(input), std::end(input), 1.0f);
    std::iota(std::begin(error), std::end(error), 1.0f);

    vector<float> expectedOutput{
        1600, 3787, 1780, 4291, 1960, 4795,
        2500, 6307, 2680, 6811, 2860, 7315,
        3400, 8827, 3580, 9331, 3760, 9835
    };
    vector<float> expectedBackOutput{
        11, 11, 35, 35, 74, 74, 69, 69, 45, 45,
        61, 61, 160, 160, 301, 301, 252, 252, 153, 153,
        168, 168, 411, 411, 735, 735, 585, 585, 342, 342,
        211, 211, 484, 484, 823, 823, 624, 624, 351, 351,
        161, 161, 359, 359, 596, 596, 441, 441, 243, 243
    };
    LayerModel model{
        convolution,
        50,
        2,
        18,
        {18, 9, 18, activation::identity},
        1,
        18,
        9,
        3,
        {5, 5, 2},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0f, 0.0f);
    internal::Convolution2D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(1))->setWeights(kernel1);
    auto output = conv.output(input, false);
    auto backOutput = conv.backOutput(input);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(Convolution, SimpleConvolution2D)
{
    auto data = createDataForConvolutionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(3, 3, 2),
        Convolution(4, 2, activation::sigmoid),
        FullyConnected(2)
    }, StochasticGradientDescent(0.05f, 0.8f));
    neuralNetwork.train(data, 3000_ep);
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