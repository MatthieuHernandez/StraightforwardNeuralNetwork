#include "../ExtendedGTest.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

TEST(Identity, WorksWithSmallNumbers)
{
    vector<vector<float>> inputData       = {{0}, {1}, {2}, {3}, {4}, {5}};
    vector<vector<float>> expectedOutputs = {{0}, {0.25}, {0.50}, {0.75}, {1.00}, {1.25}};

    Data data(regression, inputData, expectedOutputs);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), FullyConnected(4), FullyConnected(1, snn::identity)});
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

    StraightforwardNeuralNetwork neuralNetwork({Input(1), FullyConnected(4), FullyConnected(4), FullyConnected(1, snn::identity)});
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

TEST(Identity, WorksWithLotsOfNumbers)
{
    vector<vector<float>> inputData = {{9}, {2}, {7}, {5}, {1}, {8}, {6}, {3}, {4}, {0}, {9.5}, {2.5}, {7.5}, {5.5}, {1.5}, {8.5}, {6.5}, {3.5}, {4.5}, {0.5}};
    vector<vector<float>> expectedOutputs = {{18}, {4}, {14}, {10}, {2}, {16}, {12}, {6}, {8}, {0}, {19}, {5}, {15}, {11}, {3}, {17}, {13}, {7}, {9}, {1}};

    float precision = 0.4f;
    Data data(regression, inputData, expectedOutputs);
    data.setPrecision(precision);

    StraightforwardNeuralNetwork neuralNetwork({Input(1), FullyConnected(8), FullyConnected(1, snn::identity)});
    neuralNetwork.optimizer.learningRate = 0.0002;
    neuralNetwork.optimizer.momentum = 0.99;

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc || 3_s); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
    float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100
        && mae < precision
        && neuralNetwork.isValid() == 0)
        ASSERT_SUCCESS();
    else
        ASSERT_FAIL("MAE > 1: " + to_string(mae));
}