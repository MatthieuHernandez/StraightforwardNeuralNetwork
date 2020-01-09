#include "../src/neural_network/StraightforwardNeuralNetwork.hpp"
#include "../src/data/DataForClassification.hpp"

using namespace std;
using namespace chrono;
using namespace snn;

/*
This is a simple example how to use neural network for a classification problem.
The neural network return class 0 if sum of inputs is negative and class 1 il sum of input is positive.
For more explanation go to wiki.
*/
int classificationExample()
{
	vector<vector<float>> inputData = {{-0.1, 0.4, -0.6}, {0.5, -0.4, -0.8}, {-0.7, 0.9, -0.7}, {-0.9, -0.5, 1.7}, {0.5, -0.5, 0.9}, {0.3, 0.6, 0.8}};
	vector<vector<float>> expectedOutputs = {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}};

	DataForClassification data(inputData, expectedOutputs);

	StraightforwardNeuralNetwork neuralNetwork(3, {AllToAll(5), AllToAll(2)});

	neuralNetwork.trainingStart(data);
	neuralNetwork.waitFor(1.00_acc || 3_s ); // train neural network until 100% accurary or 3s on a parallel thread
	neuralNetwork.trainingStop();

	float accuracy = neuralNetwork.getGlobalClusteringRate() * 100.0f;
	int classNumber = neuralNetwork.computeCluster(data.getData(snn::testing, 0)); // consult neural network to test it
	int expectedClassNumber = data.getLabel(snn::testing, 0); // return position of neuron with highest output
	if (accuracy == 100
	&& classNumber == expectedClassNumber
	&& neuralNetwork.isValid() == 0)
	{
		return EXIT_SUCCESS; // the neural network has learned
	}
	return EXIT_FAILURE;
}