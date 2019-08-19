#include "../src/neuralNetwork/StraightforwardNeuralNetwork.h"
#include "../src/data/DataForClassification.h"
#include <thread>

using namespace std;
using namespace chrono;

/*
This is a simple example how to use neural network for a classification problem.
In this neural network return the average of 2 inputs.
For more explication go to wiki.
*/
int classificationExample()
{
	vector<vector<float>> inputData = {{-0.1, 0.8, -0.6}, {0.2, -0.4, -0.8}, {-0.7, 0.9, -0.7}, {0.9, -0.5, 0.7}, {-0.5, -0.5, 0.9}, {0.3, 0.6, 0.8}};
	vector<vector<float>> expectedOutput = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}};

	snn::DataForClassification data(inputData, expectedOutput);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 7, 2});

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(4s); // train neural network during 2 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;

	printf("accuracy = %.2f%% \n", accuracy); // Should be 100%
	int computedClass = neuralNetwork.computeCluster(inputData[0]); // consult neural network to test it

	if (computedClass - expectedOutput[0][0] < std::abs(0.3f))
		return EXIT_SUCCESS;
	return EXIT_FAILURE;
}