#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

TEST(Sequential, TestGruLayer)
{
    vector3D<float> inputData = {
        {{1.0f, 0.0f}, {0.1f, 0.0f}},
        {{-1.0f, 0.0f}, {0.1f, 0.0f}},
        {{1.0f, 0.0f}, {-0.1f, 0.0f}, {-0.2f, 0.0f}},
        {{-1.0f, 0.0f}, {-0.1f, 0.0f}, {-0.2f, 0.0f}},
        {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.1f}},
        {{-1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.1f}},
        {{1.0f, 0.0f}, {-0.1f, 0.0f}, {0.2f, 0.0f}, {0.0f, 0.2f}, {0.0f, 0.0f}},
        {{-1.0f, 0.0f}, {-0.1f, 0.0f}, {0.2f, 0.0f}, {0.0f, 0.2f}, {0.0f, 0.0f}}
    };
    vector2D<float> expectedOutputs = {
        {0, 1}, {0, 1},
        {1, 0}, {1, 0},
        {0, 1}, {0, 1}, {0, 1},
        {1, 0}, {1, 0}, {1, 0},
        {0, 1}, {0, 1}, {0, 1}, {0, 1},
        {1, 0}, {1, 0}, {1, 0}, {1, 0},
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1},
        {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}
    };
    auto data = make_unique<Data>(problem::classification, inputData, expectedOutputs, nature::sequential, 2);

    StraightforwardNeuralNetwork neuralNetwork({
        Input(2),
        GruLayer(20),
        FullyConnected(2, activation::identity, Softmax())
        },
        StochasticGradientDescent(0.003f, 0.97f));

    neuralNetwork.train(*data, 1.0_acc || 3_s, 1, 1);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0f);
}