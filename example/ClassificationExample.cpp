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
	vector<vector<float>> expectedOutputs = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}};

	snn::DataForClassification data(inputData, expectedOutputs);

	snn::StraightforwardOption option;
	snn::StraightforwardNeuralNetwork neuralNetwork({3, 5, 2});

	neuralNetwork.trainingStart(data);
	this_thread::sleep_for(4s); // train neural network during 2 seconds on parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
	int classNumber = neuralNetwork.computeCluster(data.getData(snn::testing, 0)); // consult neural network to test it
	int expectedClassNumber = data.getLabel(snn::testing, 0); // return position of neuron with highest output
	if (accuracy == 100
		&& classNumber == expectedClassNumber)
	{
		return EXIT_SUCCESS; // the neural network has learned
	}
	return EXIT_FAILURE;
}