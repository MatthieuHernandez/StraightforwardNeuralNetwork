#include <boost/serialization/smart_cast.hpp>
#include <memory>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

static auto createDataForConvolutionTests() -> Dataset;

TEST(Convolution, SimplierConvolution1D)
{
    const std::vector<float> weights{-1, 1, 1};
    std::vector<float> input{-1, 2, -3};
    const std::vector<float> expectedOutput{4, -4};
    const std::vector<float> desiredOutput{3, -3};

    StraightforwardNeuralNetwork neuralNetwork({Input(1, 3), Convolution(1, 2, activation::identity, 1)},
                                               StochasticGradientDescent(0.001F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(weights);

    auto output = neuralNetwork.computeOutput(input);

    ASSERT_EQ(output, expectedOutput);

    neuralNetwork.trainOnce(input, desiredOutput);

    auto trainedWeights = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->getWeights();

    EXPECT_NEAR(trainedWeights[0], -0.997F, 1e-6);
    EXPECT_NEAR(trainedWeights[1], 0.995F, 1e-6);
    EXPECT_NEAR(trainedWeights[2], 1.0F, 1e-6);
}

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

TEST(Convolution, SimplierConvolution2D)
{
    const std::vector<float> weights{-1, 1};
    std::vector<float> input{-2};
    const std::vector<float> expectedOutput{3};
    const std::vector<float> desiredOutput{2};

    StraightforwardNeuralNetwork neuralNetwork({Input(1, 1, 1), Convolution(1, 1, activation::identity, 1)},
                                               StochasticGradientDescent(0.001F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(weights);

    auto output = neuralNetwork.computeOutput(input);

    ASSERT_EQ(output, expectedOutput);

    neuralNetwork.trainOnce(input, desiredOutput);

    auto trainedWeights = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->getWeights();

    EXPECT_NEAR(trainedWeights[0], -0.998F, 1e-6);
    EXPECT_NEAR(trainedWeights[1], 0.999F, 1e-6);
}

TEST(Convolution, LayerConvolution2D)
{
    const std::vector<float> kernel0{1, 1, 2, 2, 3, 3, -4, -4, -5, 5, 6, -6, 7, 7, -8, -8, 9, 6, 1};
    const std::vector<float> kernel1{-10, -10, 11, -11, 12, 12, -13, 13, 14, -14, 15, -15, -16, 7, 2, 1, -8, 1, 1};
    std::vector<float> error(18);
    std::vector<float> input{1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 1, -1, 2,  -2, 3,  -3, 2,  -2, -3, 0, -3, 1, -1,
                             1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 0, 1,  -1, 2,  -2, 3,  -3, 2,  -2, 3, -3, 1, -1};
    const std::vector<float> expectedOutput{9.9908894e-01, 1.0, 9.9752736e-01, 1.0, 2.2603244e-06, 1.0,
                                            3.0590220e-07, 1.0, 9.9999970e-01, 1.0, 4.1399371e-08, 1.0,
                                            9.9330717e-01, 0.0, 1.0,           0.0, 1.0,           1.7986210e-02};
    const std::vector<float> desiredOutput{8,  159, 7,  250, -10,  175, -16, 63, 16,
                                           79, -17, 22, 0,   -140, 21,  -90, 37, -3};
    const std::vector<float> expectedkernel0{
        1.0005572, 0.9996409, 2.0007694, 1.9993628, 3.0005527, 2.9994473, -3.9995906, -4.0003433, -4.999509, 4.9994426,
        6.000769,  -6.000637, 6.999233,  7.00028,   -8.000537, -7.999591, 8.999657,   6.000491,   1.0001456};
    const std::vector<float> expectedkernel1{
        -9.999467, -10.000533, 11.000533,  -11.001066, 12.001066, 11.998401, -12.998401, 12.999467, 14.000533, -14,
        14.999467, -14.999467, -16.001066, 7.001066,   1.9984008, 1.0015992, -8.000533,  1.0005331, 0.99946696};

    StraightforwardNeuralNetwork neuralNetwork({Input(2, 5, 5), Convolution(2, 3, activation::sigmoid, 1)},
                                               StochasticGradientDescent(0.01F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->setWeights(kernel1);

    auto output = neuralNetwork.computeOutput(input);

    // ASSERT_VECTOR_EQ(output, expectedOutput, 1.0e-6F);

    neuralNetwork.trainOnce(input, desiredOutput);

    auto trainedkernel0 = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->getWeights();
    auto trainedkernel1 = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->getWeights();

    // Gradiant not equal to TensorFlow, must be reworked.
    // ASSERT_EQ(trainedkernel0, expectedkernel0);
    ASSERT_VECTOR_EQ(trainedkernel0, expectedkernel0, 1.0e-6F);
    ASSERT_VECTOR_EQ(trainedkernel1, expectedkernel1, 1.0e-6F);
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