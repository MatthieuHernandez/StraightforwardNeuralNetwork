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

	snn::StraightforwardData data(snn::regression, inputData, expectedOutput);

	snn::StraightforwardNeuralNetwork neuralNetwork(vector<int>{2, 10, 4});

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(seconds(5)); // train neural network during 5 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate();

	printf("accuracy = %.2f%%", accuracy * 100); // Should be 100%
	getchar();

	vector<float> output = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it

	if (output[0] - expectedOutput[0][0] < std::abs(0.4)
	 && output[1] - expectedOutput[0][1] < std::abs(0.4)
	 && output[2] - expectedOutput[0][2] < std::abs(0.4)
	 && output[3] - expectedOutput[0][3] < std::abs(0.4))
		return EXIT_SUCCESS; // the neural network has learned
	else
		return EXIT_FAILURE;
}
