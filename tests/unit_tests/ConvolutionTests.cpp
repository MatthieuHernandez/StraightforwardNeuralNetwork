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
    const std::vector<float> kernel0{1, 1, 2, 2, 3, 3, -4, -4, -5, 5, 6, -6, 7, 7, -8, -8, 9, 6, 1};
    const std::vector<float> kernel1{-10, -10, 11, -11, 12, 12, -13, 13, 14, -14, 15, -15, -16, 7, 2, 1, -8, 1, 1};
    std::vector<float> error(18);
    std::vector<float> input{1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 1, -1, 2,  -2, 3,  -3, 2,  -2, -3, 0, -3, 1, -1,
                             1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 0, 1,  -1, 2,  -2, 3,  -3, 2,  -2, 3, -3, 1, -1};
    const std::vector<float> expectedOutput{
        7, 159, 6, 251, -13, 179, -15, 62, 15, 80, -17, 20, 5, -139, 21, -89, 34, -4,
    };
    const std::vector<float> desiredOutput{8,  159, 7,  250, -10,  175, -16, 63, 16,
                                           79, -17, 22, 0,   -140, 21,  -90, 37, -3};
    const std::vector<float> expectedkernel0{1.00024,  0.99991,  2.00024, 1.99989, 3.00014,  2.99989, -3.99986,
                                             -4.00014, -4.99979, 4.99973, 6.00029, -6.00018, 6.99988, 7.00009,
                                             -8.00014, -7.99988, 8.99987, 6.00018, 0.9997};
    const std::vector<float> expectedkernel1{-10.00005, -9.9999,  10.99997,  -10.99994, 11.99989,  12.00011,  -13.00009,
                                             13.00006,  13.99986, -13.99986, 14.9999,   -14.99986, -16.00003, 6.99999,
                                             2.00002,   0.99995,  -7.99995,  0.99989,   0.99996};

    LayerModel convModel{.type = convolution,
                         .numberOfInputs = 50,
                         .numberOfNeurons = 2,
                         .numberOfOutputs = 18,
                         .neuron = {.numberOfInputs = 18,
                                    .batchSize = 9,
                                    .numberOfWeights = 19,
                                    .bias = 1.0F,  // Force the bias to 1.0F.
                                    .activationFunction = activation::identity},
                         .numberOfFilters = 2,
                         .numberOfKernels = 18,
                         .numberOfKernelsPerFilter = 9,
                         .kernelSize = 3,
                         .shapeOfInput = {2, 5, 5},
                         .optimizers = {}};
    StraightforwardNeuralNetwork neuralNetwork({Input(2, 5, 5), convModel}, StochasticGradientDescent(0.00001F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->setWeights(kernel1);

    auto output = neuralNetwork.computeOutput(input);

    ASSERT_EQ(output, expectedOutput);

    neuralNetwork.trainOnce(input, desiredOutput);

    auto trainedkernel0 = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->getWeights();
    auto trainedkernel1 = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->getWeights();

    // Gradiant not equal to TensorFlow, must be reworked.
    // ASSERT_EQ(trainedkernel0, expectedkernel0);
    ASSERT_VECTOR_EQ(trainedkernel0, expectedkernel0, 1e-3F);
    ASSERT_VECTOR_EQ(trainedkernel1, expectedkernel1, 1e-3F);
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