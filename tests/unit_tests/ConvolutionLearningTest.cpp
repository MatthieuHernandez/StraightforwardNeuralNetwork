#include <boost/serialization/smart_cast.hpp>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

TEST(ConvolutionLearning, LayerConvolution2D)
{
    const int size = 49;
    vector2D<float> inputs(size, std::vector<float>(size, -0.5F));
    for (int i = 0; i < size; ++i)
    {
        inputs[i][i] = 0.5F;
    }
    Dataset dataset(problem::regression, inputs, inputs);
    dataset.setPrecision(0.1F);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1, 7, 7), Convolution(1, 1, activation::tanh), Convolution(49, 7, activation::tanh)},
        StochasticGradientDescent(0.02F, 0.0F));
    neuralNetwork.train(dataset, 500_ep /*|| 1.0_acc*/);
    const float accuracy = neuralNetwork.getGlobalClusteringRate();

    std::cout << neuralNetwork.summary() << '\n';

    // auto toto = neuralNetwork.computeOutput(inputs[12]);

    ASSERT_ACCURACY(accuracy, 1.0);
}
