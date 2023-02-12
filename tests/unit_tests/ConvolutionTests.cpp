#include <memory>
#include <numeric>
#include <boost/serialization/smart_cast.hpp>

#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

Data createDataForConvolutionTests();

TEST(Convolution, LayerConvolution1D)
{
    vector<float> input {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<float> kernel0 {1, 2, 3, 4, 5, 6};
    vector<float> kernel1 {7, 8, 9, 10, 11, 12};
    vector<float> error {1, 2, 3, 4, 5, 6};

    vector<float> expectedOutput {92, 218, 134, 332, 176, 446};
    vector<float> expectedBackOutput {15, 18, 52, 62, 119, 140, 128, 146, 91, 102};
    LayerModel model {
        convolution,
        10,
        2,
        6,
        {6, 3, 6, 1.0f, activation::identity},
        2,
        6,
        3,
        3,
        {2, 5},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0f, 0.0f);
    internal::Convolution1D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(1))->setWeights(kernel1);
    auto output = conv.output(input, false);
    auto backOutput = conv.backOutput(input);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(Convolution, LayerConvolution2D)
{
    vector<float> input(50);
    vector<float> kernel0 {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
    vector<float> kernel1 {10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18};
    vector<float> error(18);
    std::iota(std::begin(input), std::end(input), 1.0f);
    std::iota(std::begin(error), std::end(error), 1.0f);

    vector<float> expectedOutput {
        1600, 3787, 1780, 4291, 1960, 4795,
        2500, 6307, 2680, 6811, 2860, 7315,
        3400, 8827, 3580, 9331, 3760, 9835
    };
    vector<float> expectedBackOutput {
        21, 21, 67, 67, 142, 142, 133, 133, 87, 87,
        117, 117, 308, 308, 581, 581, 488, 488, 297, 297,
        324, 324, 795, 795, 1425, 1425, 1137, 1137, 666, 666,
        411, 411, 944, 944, 1607, 1607, 1220, 1220, 687, 687,
        315, 315, 703, 703, 1168, 1168, 865, 865, 477, 477

    };
    LayerModel model {
        convolution,
        50,
        2,
        18,
        {18, 9, 18, 1.0f, activation::identity},
        1,
        18,
        9,
        3,
        {2, 5, 5},
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

TEST(Convolution, Momentum)
{
    vector<float> input {1.0f, 2.0f, 3.0f, 4.0f};

    LayerModel model{
        convolution,
        4,
        1,
        4,
        {1, 4, 1, 1.0f, activation::identity},
        1,
        4,
        4,
        1,
        {1, 2, 2},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.1f, 0.9f);
    internal::Convolution2D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights({1.0f});
  

    for (auto i = 0; i < 3; ++i)
    {
        vector<float> error {1.0f, 2.0f, 3.0f, 4.0f};
        auto output = conv.output(input, false);
        auto backOutput = conv.backOutput(error);
        ASSERT_GT(output.size(), 0);
        ASSERT_GT(backOutput.size(), 0);
    }
}

TEST(Convolution, SimpleConvolution1D)
{
    auto data = createDataForConvolutionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(2, 9),
            Convolution(3, 4, activation::iSigmoid),
            FullyConnected(2)
        },
        StochasticGradientDescent(0.03f, 0.8f));
    neuralNetwork.train(data, 3000_ep);
    float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}

TEST(Convolution, SimpleConvolution2D)
{
    auto data = createDataForConvolutionTests();
    StraightforwardNeuralNetwork neuralNetwork({
        Input(2, 3, 3),
        Convolution(4, 2, activation::iSigmoid),
        FullyConnected(2)
    }, StochasticGradientDescent(0.03f, 0.8f));
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