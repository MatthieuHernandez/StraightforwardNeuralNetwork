#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "neural_network/layer/perceptron/activation_function/Tanh.hpp"

using namespace std;
using namespace snn;

TEST(Identity, WorksWithSmallNumbers)
{
    vector<vector<float>> inputData       = {{0}, {1}, {2}, {3}, {4}, {5}};
    vector<vector<float>> expectedOutputs = {{0}, {0.25}, {0.50}, {0.75}, {1.00}, {1.25}};

    Data data(regression, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), AllToAll(4), AllToAll(1, snn::identity)});
    neuralNetwork.optimizer.learningRate = 0.02;
    neuralNetwork.optimizer.momentum = 0.99;

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(0.01_mae || 3_s);
    neuralNetwork.stopTraining();

    float mae = neuralNetwork.getMeanAbsoluteError();

    if (mae <= 0.01)
        ASSERT_SUCCESS();
    else
        ASSERT_FAIL("MAE > 1: " + to_string(mae));
}

TEST(Identity, WorksWithBigNumbers)
{
    vector<vector<float>> inputData       = {{0}, {1}, {2}, {3}, {4}, {5}, {6}};
    vector<vector<float>> expectedOutputs = {{0}, {25}, {50}, {75}, {100}, {125}, {150}};

    Data data(regression, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), AllToAll(4), AllToAll(4), AllToAll(1, snn::identity)});
    neuralNetwork.optimizer.learningRate = 0.0001;
    neuralNetwork.optimizer.momentum = 0.99;

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.0_mae || 3_s);
    neuralNetwork.stopTraining();

    float mae = neuralNetwork.getMeanAbsoluteError();

    if (mae <= 1)
        ASSERT_SUCCESS();
    else
        ASSERT_FAIL("MAE > 1: " + to_string(mae));
}