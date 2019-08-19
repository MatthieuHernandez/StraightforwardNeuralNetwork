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
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 8, 1}, {tanH, tanH}, option);

		vector<float> output1 = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it
		vector<float> output2 = neuralNetwork.computeOutput(inputData[1]); // consult neural network to test it
		vector<float> output3 = neuralNetwork.computeOutput(inputData[2]); // consult neural network to test it
		vector<float> output4 = neuralNetwork.computeOutput(inputData[3]); // consult neural network to test it
		printf("output1 = %.4f \n", output1[0]);
		printf("output2 = %.4f \n", output2[0]);
		printf("output3 = %.4f \n", output3[0]);
		printf("output4 = %.4f \n\n", output4[0]);

	while(true)
	{
		neuralNetwork.trainingStart(data);
		this_thread::sleep_for(8s); // train neural network during 4 seconds on parallel thread
		neuralNetwork.trainingStop();
		this_thread::sleep_for(1s);
		float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;

		printf("accuracy = %.2f%% \n\n", accuracy); // Should be 100%
		vector<float> output1 = neuralNetwork.computeOutput(inputData[0]); // consult neural network to test it
		vector<float> output2 = neuralNetwork.computeOutput(inputData[1]); // consult neural network to test it
		vector<float> output3 = neuralNetwork.computeOutput(inputData[2]); // consult neural network to test it
		vector<float> output4 = neuralNetwork.computeOutput(inputData[3]); // consult neural network to test it
		printf("output1 = %.4f \n", output1[0]);
		printf("output2 = %.4f \n", output2[0]);
		printf("output3 = %.4f \n", output3[0]);
		printf("output4 = %.4f \n\n", output4[0]);
		if (accuracy > 0.9)
		{
			//return EXIT_SUCCESS; // the neural network has learned
		}
		//return EXIT_FAILURE;
	}
}
