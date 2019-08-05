#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include <thread>

using namespace std;
using namespace chrono;

/*
This is the simplest example how to use this library
In this neural network return the average of 2 inputs.
For more explication go to wiki
*/
int simpleExampleClassification1()
{
	vector<vector<float>> inputData = {{0, 0, 0}, {1, 1, 1}};
	vector<vector<float>> expectedOutput = {{0}, {1}};

	snn::StraightforwardData data(snn::regression, inputData, expectedOutput);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 7, 1});

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(2s); // train neural network during 2 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;

	printf("accuracy = %.2f%%", accuracy); // Should be 100%
	vector<float> output = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it

	if (output[0] - expectedOutput[0][0] < std::abs(0.3f))
		return EXIT_SUCCESS;
	return EXIT_FAILURE;
}

int simpleExampleClassification2()
{
	vector<vector<float>> inputData = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {1, 1, 1}};
	vector<vector<float>> expectedOutput = {{0.333}, {0.666}, {0}, {1}};

	snn::StraightforwardData data(snn::regression, inputData, expectedOutput);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 8, 1}, {sigmoid, sigmoid}, option);

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(4s); // train neural network during 4 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;

	printf("accuracy = %.2f%%", accuracy); // Should be 100%
	vector<float> output = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it

	if (output[0] - 0.333 < std::abs(0.1f))
	 {
		 printf("WHAT ???");
		return EXIT_SUCCESS; // the neural network has learned
	 }
	else
	{

		printf("\n%.2f - %.2f = %.2f\n", output[0], expectedOutput[0][0], std::abs(0.3f));
		return EXIT_FAILURE;
	}
}
