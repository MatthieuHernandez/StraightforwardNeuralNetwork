#include "StraightforwardNeuralNetwork.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
using namespace snn;

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork)
	: NeuralNetwork(structureOfNetwork,
	                std::vector<activationFunctionType>(structureOfNetwork.size() - 1, sigmoid),
	                0.05f,
	                0.0f)
{
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
                                                           const std::vector<activationFunctionType>&
                                                           activationFunctionByLayer,
                                                           float learningRate,
                                                           float momentum)
	: NeuralNetwork(structureOfNetwork,
	                activationFunctionByLayer,
	                learningRate,
	                learningRate)
{
}

std::vector<float> StraightforwardNeuralNetwork::computeOutput(std::vector<float> inputs)
{
	throw std::exception();
}

int StraightforwardNeuralNetwork::computeCluster(std::vector<float> inputs)
{
	throw std::exception();
}

void StraightforwardNeuralNetwork::trainingStart(StraightforwardData data)
{
	throw std::exception();
}


void StraightforwardNeuralNetwork::trainingStop()
{
	throw std::exception();
}
