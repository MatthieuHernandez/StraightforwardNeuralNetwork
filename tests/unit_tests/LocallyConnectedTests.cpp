#include <memory>
#include <numeric>
#include <boost/serialization/smart_cast.hpp>

#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

Data createDataForLocallyConnectedTests();

TEST(LocallyConnected, LayerLocallyConnected1D)
{
    vector<float> input {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<float> kernel0 {1, 3, 5, 7, 9, 11};
    vector<float> kernel1 {2, 4, 6, 8, 10, 12};
    vector<float> kernel2 {13, 15, 17, 19, 21, 23};
    vector<float> kernel3 {14, 16, 18, 20, 22, 24};
    vector<float> error {1, 2, 3, 4};

    vector<float> expectedOutput {162, 183, 555, 589};
    vector<float> expectedBackOutput {40, 48, 56, 64, 72, 80, 60, 72, 84, 96};
    LayerModel model {
        locallyConnected,
        10,
        4,
        5,
        {6, 1, 6, 1.0f, activation::identity},
        2,
        4,
        2,
        3,
        {2, 5},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0f, 0.0f);
    internal::LocallyConnected1D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights(kernel0);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(1))->setWeights(kernel1);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(2))->setWeights(kernel2);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(3))->setWeights(kernel3);
    auto output = conv.output(input, false);
    auto backOutput = conv.backOutput(input);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(LocallyConnected, SimpleLayerLocallyConnected2D)
{
    vector<float> input {1, 2, 3, 4};
    vector<float> error(9);
    std::iota(std::begin(error), std::end(error), 1.0f);

    vector<float> expectedOutput {2, 5, 10, 17};
    vector<float> expectedBackOutput {1, 4, 9, 16};
    LayerModel model {
        locallyConnected,
        4,
        4,
        4,
        {1, 1, 1, 1.0f, activation::identity},
        1,
        4,
        4,
        1,
        {1, 2, 2},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0f, 0.0f);
    internal::LocallyConnected2D conv(model, sgd);
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(0))->setWeights({1});
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(1))->setWeights({2});
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(2))->setWeights({3});
    static_cast<internal::SimpleNeuron*>(conv.getNeuron(3))->setWeights({4});
    auto output = conv.output(input, false);
    auto backOutput = conv.backOutput(input);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(LocallyConnected, ComplexeLayerLocallyConnected2D)
{
    vector<float> input(50);
    vector<vector<float>> kernels {
                              {1., 3., 5., 7., 9., 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35},
                              {2., 4., 6., 8., 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36},
                              {37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71},
                              {38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72},
                              {73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107},
                              {74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108},
                              {109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143},
                              {110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144},

};
    vector<float> error(8);
    std::iota(std::begin(input), std::end(input), 1.0f);
    std::iota(std::begin(error), std::end(error), 1.0f);

    vector<float> expectedOutput {
        5920, 6163, 12535, 12757,
        39239, 39701, 41325, 41673
    };
    vector<float> expectedBackOutput {
        5, 11, 17, 23, 29, 35, 263, 277, 291, 305,
        41, 47, 53, 59, 65, 71, 347, 361, 375, 389,
        77, 83, 89, 95, 101, 107, 431, 445, 459, 473,
        809, 831, 853, 875, 897, 919, 1643, 1673, 1703, 1733,
        941, 963, 985, 1007, 1029, 1051, 1823, 1853, 1883, 1913
    };
    LayerModel model{
        locallyConnected,
        50,
        8,
        18,
        {18, 1, 18, 1.0f, activation::identity},
        2,
        8,
        4,
        3,
        {2, 5, 5},
        {}
    };
    auto sgd = std::make_shared<internal::StochasticGradientDescent>(0.0f, 0.0f);
    internal::LocallyConnected2D conv(model, sgd);
    for(int k = 0; k < 8; ++k)
        static_cast<internal::SimpleNeuron*>(conv.getNeuron(k))->setWeights(kernels[k]);
    auto output = conv.output(input, false);
    auto backOutput = conv.backOutput(input);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(LocallyConnected, SimpleLocallyConnected1D)
{
    auto data = createDataForLocallyConnectedTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(2, 9),
            LocallyConnected(4, 2, activation::sigmoid),
            FullyConnected(2)
        },
        StochasticGradientDescent(0.05f, 0.1f));
    neuralNetwork.train(data, 3000_ep);
    float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}

TEST(LocallyConnected, SimpleLocallyConnected2D)
{
    auto data = createDataForLocallyConnectedTests();
    StraightforwardNeuralNetwork neuralNetwork(
        {
            Input(2, 3, 3),
            LocallyConnected(4, 2, activation::sigmoid),
            FullyConnected(2)
        },
        StochasticGradientDescent(0.05f, 0.1f));
    neuralNetwork.train(data, 3000_ep);
    float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0);
}

Data createDataForLocallyConnectedTests()
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
    vector2D<float> expectedOutputs = { {0, 1}, {0, 1}, {1, 0} };

    return Data(problem::classification, inputData, expectedOutputs);
}