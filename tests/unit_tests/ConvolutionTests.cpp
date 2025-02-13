#include <boost/serialization/smart_cast.hpp>
#include <memory>
#include <numeric>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

static auto createDataForConvolutionTests() -> Dataset;

TEST(Convolution, LayerConvolution1D)
{
    std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::vector<float> kernel0{1, 2, 3, 4, 5, 6, 1};
    const std::vector<float> kernel1{7, 8, 9, 10, 11, 12, 1};
    const std::vector<float> error{1, 2, 3, 4, 5, 6};

    const std::vector<float> expectedOutput{92, 218, 134, 332, 176, 446};
    const std::vector<float> expectedBackOutput{15, 18, 52, 62, 119, 140, 128, 146, 91, 102};
    LayerModel model{convolution, 10, 2, 6, {6, 3, 7, 1.0F, activation::identity}, 2, 6, 3, 3, {2, 5}, {}};
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0F, 0.0F);
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
    std::vector<float> input(50);
    const std::vector<float> kernel0{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 1};
    const std::vector<float> kernel1{10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 1};
    std::vector<float> error(18);
    std::iota(std::begin(input), std::end(input), 1.0F);
    std::iota(std::begin(error), std::end(error), 1.0F);

    const std::vector<float> expectedOutput{1600, 3787, 1780, 4291, 1960, 4795, 2500, 6307, 2680,
                                            6811, 2860, 7315, 3400, 8827, 3580, 9331, 3760, 9835};
    const std::vector<float> expectedBackOutput{
        21,   21,   67,   67,   142, 142, 133, 133,  87,   87,   117,  117,  308, 308, 581, 581, 488,
        488,  297,  297,  324,  324, 795, 795, 1425, 1425, 1137, 1137, 666,  666, 411, 411, 944, 944,
        1607, 1607, 1220, 1220, 687, 687, 315, 315,  703,  703,  1168, 1168, 865, 865, 477, 477};
    LayerModel model{convolution, 50, 2, 18, {18, 9, 19, 1.0F, activation::identity}, 1, 18, 9, 3, {2, 5, 5}, {}};
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0F, 0.0F);
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
    const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};

    LayerModel model{convolution, 4, 1, 4, {1, 4, 2, 1.0F, activation::identity}, 1, 4, 4, 1, {1, 2, 2}, {}};
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.1F, 0.9F);
    internal::Convolution2D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights({1.0F, 0.0F});

    for (auto i = 0; i < 3; ++i)
    {
        std::vector<float> error{1.0F, 2.0F, 3.0F, 4.0F};
        auto output = conv.output(input, false);
        auto backOutput = conv.backOutput(error);
        ASSERT_GT(output.size(), 0);
        ASSERT_GT(backOutput.size(), 0);
    }
}

TEST(Convolution, SimpleConvolution1D)
{
    auto dataset = createDataForConvolutionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2, 9), Convolution(3, 4, activation::iSigmoid), FullyConnected(2)},
        StochasticGradientDescent(0.03F, 0.8F));
    neuralNetwork.train(dataset, 3000_ep);
    const float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}

TEST(Convolution, SimpleConvolution2D)
{
    auto dataset = createDataForConvolutionTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2, 3, 3), Convolution(4, 2, activation::iSigmoid), FullyConnected(2)},
        StochasticGradientDescent(0.03F, 0.8F));
    neuralNetwork.train(dataset, 3000_ep);
    const float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}

auto createDataForConvolutionTests() -> Dataset
{
    vector2D<float> inputData = {
        {-1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
         -1.0F, -1.0F},

        {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F},

        {0.11F, 0.12F, 0.21F, 0.22F, 0.31F, 0.32F, 0.41F, 0.42F, 0.51F, 0.52F, 0.61F, 0.62F, 0.71F, 0.72F, 0.81F, 0.82F,
         0.91F, 0.92F}};
    vector2D<float> expectedOutputs = {{0, 1}, {0, 1}, {1, 0}};

    return {problem::classification, inputData, expectedOutputs};
}