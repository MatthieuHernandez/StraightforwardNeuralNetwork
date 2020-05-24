#include "Examples.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a time series.
In this neural network return the sum of 2 inputs.
For more explanation go to wiki.
*/
int recurrenceExample()
{
    vector<vector<float>> inputData       = {{3}, {0}, {3}, {5}, {4}, {2}, {0}, {2}, {4}, {1}, {4}, {3}, {0}, {4}, {3}, {2}, {1}, {2}, {0}, {1}, {5}, {5}};
    vector<vector<float>> expectedOutputs = {{3}, {3}, {3}, {8}, {9}, {6}, {2}, {2}, {6}, {5}, {5}, {7}, {3}, {4}, {7}, {5}, {3}, {3}, {2}, {1}, {6}, {10}};

    float precision = 0.5f;
    Data data(regression, inputData, expectedOutputs, continuous, 1);
    data.setPrecision(precision);

    snn::StraightforwardNeuralNetwork neuralNetwork({Input(1), Recurrence(3, 2), AllToAll(1, snn::identity)});

    neuralNetwork.startTraining(data);
    neuralNetwork.waitFor(1.00_acc || 10_s ); // train neural network until 100% accurary or 3s on a parallel thread
    neuralNetwork.stopTraining();

    float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
    float mae = neuralNetwork.getMeanAbsoluteError();

    if (accuracy == 100
        && mae < 0.1
        && neuralNetwork.isValid() == 0)
    {
        return EXIT_SUCCESS; // the neural network has learned
    }
    return EXIT_FAILURE;
}