#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include "../src/data/DataForRegression.h"
#include <thread>

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a regression.
In this neural network return the average of 3 inputs.
For more explication go to wiki.
*/
int regressionExample()
{
	vector<vector<float>> inputData = {{0, 1, 0}, {0, 1, 1},{1, 0, 1}, {1, 1, 0}, {0, 0, 0}, {1, 1, 1}};
	vector<vector<float>> expectedOutputs = {{0.333}, {0.666}, {0.666}, {0.666}, {0}, {1}};

	float precision = 0.1f;
	snn::DataForRegression data(inputData, expectedOutputs, precision);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 5, 1}, {sigmoid, sigmoid}, option);

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(2s); // train neural network during 2 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
	vector<float> output = neuralNetwork.computeOutput(data.getData(snn::testing, 0)); // consult neural network to test it
	vector<float> expectedOutput = data.getOutputs(snn::testing, 0);

	if (accuracy == 100
		&& abs(output[0] - expectedOutput[0]) < precision)
	{
		return EXIT_SUCCESS; // the neural network has learned
	}
	return EXIT_FAILURE;
}