#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include <thread>

using namespace std;
using namespace chrono;

/*
* This is the simpliest example how to use this library
* In this neural network return 3 ouputs AND, NAND, OR, XOR logical operator of 2 inputs.
* For more explaination go to wiki
*/
int main()
{
	vector<vector<float>> inputData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	vector<vector<float>> expectedOutput = {{0, 1, 0, 0}, {0, 1, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 0}};

	snn::StraightforwardData data(regression, inputData, expectedOutput);

	snn::StraightforwardNeuralNetwork neuralNetwork(vector<int>{2, 10, 10, 4});

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(seconds(15)); // train neural network during 15 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate();

	printf("accuracy = %.2f", accuracy);
	getchar();

	vector<float> output = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it

	if (output == expectedOutput[0])
		return EXIT_SUCCESS; // the neural network has learned
	else
		return EXIT_FAILURE;
}