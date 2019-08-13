#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include "../src/data/DataForRegression.h"
#include <thread>

using namespace std;
using namespace chrono;

/*
This is a simple example how to use neural network for a regression.
In this neural network return the average of 3 inputs.
For more explication go to wiki.
*/
int regressionExample()
{
	vector<vector<float>> inputData = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {1, 1, 1}};
	vector<vector<float>> expectedOutput = {{0.333}, {0.666}, {0}, {1}};

	float precision = 0.1f;
	snn::DataForRegression data(inputData, expectedOutput, precision);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 8, 1}, {tanH, sigmoid}, option);

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(4s); // train neural network during 4 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;

	printf("accuracy = %.2f%%", accuracy); // Should be 100%
	vector<float> output = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it

	if (std::abs(output[0] - expectedOutput[0][0]) < precision)
	 {
		return EXIT_SUCCESS; // the neural network has learned
	 }
	return EXIT_FAILURE;
}
