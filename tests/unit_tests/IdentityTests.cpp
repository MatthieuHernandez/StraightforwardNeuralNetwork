#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

TEST(Identity, WorksWithSmallNumbers)
{
    vector2D<float> inputData = {{0}, {1}, {2}, {3}, {4}, {5}};
    vector2D<float> expectedOutputs = {{0}, {0.25}, {0.50}, {0.75}, {1.00}, {1.25}};

    Data data(problem::regression, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), FullyConnected(6), FullyConnected(1, snn::activation::identity)},
        StochasticGradientDescent(0.02F, 0.99F));

    neuralNetwork.train(data, 0.01_mae || 2_s);

    float mae = neuralNetwork.getMeanAbsoluteErrorMin();

    if (mae <= 0.01)
        ASSERT_SUCCESS();
    else
        ASSERT_FAIL("MAE > 1: " + to_string(mae));
}

TEST(Identity, WorksWithBigNumbers)
{
    vector2D<float> inputData = {{-3}, {-2}, {-1}, {0}, {1}, {2}, {3}, {4}};
    vector2D<float> expectedOutputs = {{-75}, {-50}, {-25}, {0}, {25}, {50}, {75}, {100}};

    Data data(problem::regression, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork(
        {Input(1), FullyConnected(20), FullyConnected(8), FullyConnected(4, activation::tanh),
         FullyConnected(1, activation::identity)},
        StochasticGradientDescent(0.00001F, 0.95F));

    neuralNetwork.train(data, 1.0_mae || 3_s);

    float mae = neuralNetwork.getMeanAbsoluteErrorMin();

    ASSERT_MAE(mae, 2);
}

TEST(Identity, WorksWithLotsOfNumbers)
{
    vector2D<float> inputData = {{9},   {2},   {7},   {5},   {1},   {8},   {6},   {3},   {4},   {0},
                                 {9.5}, {2.5}, {7.5}, {5.5}, {1.5}, {8.5}, {6.5}, {3.5}, {4.5}, {0.5}};
    vector2D<float> expectedOutputs = {{18}, {4}, {14}, {10}, {2}, {16}, {12}, {6}, {8}, {0},
                                       {19}, {5}, {15}, {11}, {3}, {17}, {13}, {7}, {9}, {1}};

    float precision = 0.4F;
    Data data(problem::regression, inputData, expectedOutputs);
    data.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), FullyConnected(30), FullyConnected(1, activation::identity)},
                                               StochasticGradientDescent(0.001F, 0.80F));

    neuralNetwork.train(data, 1.00_acc || 2_s);

    float accuracy = neuralNetwork.getGlobalClusteringRateMax() * 100.0F;
    float mae = neuralNetwork.getMeanAbsoluteErrorMin();

    if (accuracy == 100 && mae < precision && neuralNetwork.isValid() == ErrorType::noError)
        ASSERT_SUCCESS();
    else
        ASSERT_FAIL("MAE > 0.4: " + to_string(mae));
}