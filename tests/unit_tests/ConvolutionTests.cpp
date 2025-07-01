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
    const std::vector<float> expectedOutput{4, -4, 4};
    const std::vector<float> desiredOutput{3, -3, 3};

    StraightforwardNeuralNetwork neuralNetwork({Input(1, 3), Convolution(1, 2, activation::identity, 1)},
                                               StochasticGradientDescent(0.001F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(weights);

    auto output = neuralNetwork.computeOutput(input);

    ASSERT_EQ(output, expectedOutput);

    neuralNetwork.trainOnce(input, desiredOutput);

    auto trainedWeights = static_cast<internal::Neuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->getWeights();

    EXPECT_NEAR(trainedWeights[0], -0.994F, 1e-6);
    EXPECT_NEAR(trainedWeights[1], 0.995F, 1e-6);
    EXPECT_NEAR(trainedWeights[2], 0.999F, 1e-6);
}

TEST(Convolution, LayerConvolution1D)
{
    std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::vector<float> kernel0{1, 2, 3, 4, 5, 6, 1};
    const std::vector<float> kernel1{7, 8, 9, 10, 11, 12, 1};
    const std::vector<float> error{1, 2, 3, 4, 5, 6};

    const std::vector<float> expectedOutput{51, 111, 92, 218, 134, 332, 176, 446, 91, 295};
    const std::vector<float> desiredOutput{50, 110, 93, 219, 134, 330, 178, 445, 92, 294};
    const std::vector<float> expectedTrainedOutput{53.32,  108.57, 96.84, 213.06, 141.54,
                                                   324.38, 186.24, 435.7, 99.1,   287.21};
    StraightforwardNeuralNetwork neuralNetwork({Input(2, 5), Convolution(2, 3, activation::identity, 1.0F)},
                                               StochasticGradientDescent(0.01F, 0.0F));
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0F, 0.0F);
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->setWeights(kernel1);

    auto output = neuralNetwork.computeOutput(input, false);
    ASSERT_EQ(output, expectedOutput);

    neuralNetwork.trainOnce(input, desiredOutput);
    auto trainedOutput = neuralNetwork.computeOutput(input, false);
    ASSERT_VECTOR_EQ(trainedOutput, expectedTrainedOutput, 1.0e-4F);
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
    const std::vector<float> kernel0{0.1, 0.1,  0.2, 0.2, 0.3,  0.3,  -0.4, -0.4, -0.1, 0.3,
                                     0.6, -0.2, 0.1, 0.1, -0.4, -0.8, 0.5,  0.3,  1};
    const std::vector<float> kernel1{0.0, -0.1, 0.1,  -0.1, 0.2, 0.2, -0.3, 0.3, 0.4, -0.4,
                                     0.5, -0.5, -0.6, -0.2, 2,   1,   -0.4, 0.1, 1};

    std::vector<float> input{1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 1, -1, 2,  -2, 3,  -3, 2,  -2, -3, 0, -3, 1, -1,
                             1, -1, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 0, 1,  -1, 2,  -2, 3,  -3, 2,  -2, 3, -3, 1, -1};
    const std::vector<float> expectedOutput{
        0.99263155, 0.9996318,  0.9977749,  0.9997978,  0.9926315,  0.9996318,  0.99955034,  0.99990916, 0.53704965,
        0.9950547,  0.8853517,  -0.935409,  0.9704519,  0.7615942,  0.970452,   0.99999833,  0.09966807, 0.9999945,
        -0.8617232, 0.6640369,  -0.8336545, -0.9998891, -0.716298,  0.8617232,  0.8336547,   -0.998508,  0.2913126,
        -0.9757431, 0.9640276,  0.97574323, 0.2913127,  -0.9987782, 0.8853516,  -0.60436803, 0.970452,   -0.379949,
        0.9997978,  0.9996318,  0.935409,   0.8336546,  0.99100745, 0.99995035, 0.71629786,  0.9959493,  0.9640276,
        0.9939632,  0.19737516, 0.9950547,  0.4621171,  0.19737512};
    const std::vector<float> desiredOutput{0.9,  0.5, 0.7,  0.3, 0.8,  0.5, -0.3, -0.6, 1.0, 0.9, -0.7, 0.2, 0.0,
                                           -0.4, 1.0, -0.9, 0.3, -0.1, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0, 0.0,
                                           8.0,  0.5, 0.7,  0.3, -0.1, 0.7, -0.6, 0.6,  0.6, 0.7, -0.1, 0.3, 0.2,
                                           -0.4, 0.2, -0.9, 0.3, -0.3, 0.2, 0.2,  0.0,  0.0, 0.0, -1.0};
    const std::vector<float> expectedkernel0{0.27902007, -0.06096088, 0.44511837,  -0.02778675, 0.45960167,
                                             0.15179253, -0.46600983, -0.33563125, -0.16942912, 0.42598987,
                                             0.45864967, 0.00732283,  -0.11940194, 0.17634685,  -0.48385707,
                                             -0.7650452, 0.546436,    0.26052827,  1.0611302};
    const std::vector<float> expectedkernel1{-0.04862903, -0.08483574, 0.04416766, -0.06316294, 0.16858868,
                                             0.2408669,   -0.38272187, 0.39436263, 0.33641678,  -0.34247598,
                                             0.45953518,  -0.48329076, -0.5451834, -0.24273613, 2.0587895,
                                             0.9400093,   -0.34894758, 0.05051783, 0.99235386};

    StraightforwardNeuralNetwork neuralNetwork({Input(2, 5, 5), Convolution(2, 3, activation::tanh, 1)},
                                               StochasticGradientDescent(0.01F, 0.0F));
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(neuralNetwork.layers.at(0)->getNeuron(1))->setWeights(kernel1);

    auto output = neuralNetwork.computeOutput(input);

    ASSERT_VECTOR_EQ(output, expectedOutput, 1.0e-6F);

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