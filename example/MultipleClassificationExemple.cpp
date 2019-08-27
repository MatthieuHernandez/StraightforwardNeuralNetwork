#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include "../src/data/DataForMultipleClassification.h"
#include <thread>

using namespace std;
using namespace chrono;

/*
This is the simplest example how to use this library
In this neural network return 3 ouputs AND, NAND, OR logical operator of 2 inputs.
For more explication go to wiki
*/
int multipleClassificationExample()
{
	vector<vector<float>> inputData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	vector<vector<float>> expectedOutputs = {{0, 1, 0}, {0, 1, 1}, {0, 1, 1}, {1, 0, 1}};

	float separator = 0.5f;
	snn::DataForMultipleClassification data(inputData, expectedOutputs, separator);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({2, 8, 3}, {sigmoid, sigmoid}, option);

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(4s); // train neural network during 4 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
	vector<float> output = neuralNetwork.computeOutput(data.getData(snn::testing, 0)); // consult neural network to test it

	if (accuracy == 100
		&& output[0] < separator
		&& output[1] > separator
		&& output[2] < separator)
	{
		return EXIT_SUCCESS; // the neural network has learned
	}
	return EXIT_FAILURE;
}
