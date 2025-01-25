#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

TEST(Sequential, TestGruLayer)
{
    vector3D<float> inputData = {{{1.0F, 0.0F}, {0.1F, 0.0F}},
                                 {{-1.0F, 0.0F}, {0.1F, 0.0F}},
                                 {{1.0F, 0.0F}, {-0.1F, 0.0F}, {-0.2F, 0.0F}},
                                 {{-1.0F, 0.0F}, {-0.1F, 0.0F}, {-0.2F, 0.0F}},
                                 {{1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.1F}},
                                 {{-1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.1F}},
                                 {{1.0F, 0.0F}, {-0.1F, 0.0F}, {0.2F, 0.0F}, {0.0F, 0.2F}, {0.0F, 0.0F}},
                                 {{-1.0F, 0.0F}, {-0.1F, 0.0F}, {0.2F, 0.0F}, {0.0F, 0.2F}, {0.0F, 0.0F}}};
    vector2D<float> expectedOutputs = {{0, 1}, {0, 1}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0},
                                       {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1},
                                       {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}};
    auto data = std::make_unique<Data>(problem::classification, inputData, expectedOutputs, nature::sequential, 2);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(2), GruLayer(20), FullyConnected(2, activation::identity, Softmax())},
        StochasticGradientDescent(0.003F, 0.97F));

    neuralNetwork.train(*data, 1.0_acc || 3_s, 1, 1);

    auto accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0F);
}