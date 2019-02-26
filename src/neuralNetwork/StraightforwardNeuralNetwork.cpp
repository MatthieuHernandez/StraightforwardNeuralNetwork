#include "StraightforwardNeuralNetwork.h"
#include "neuralNetwork/layer/perceptron/activationFunction/activationFunction.h"
using namespace snn;

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork)
{
	float learningRate = 0.05f;
	float momentum = 0.0f;
	std::vector<activationFunctionType> activationFunctionByLayer(structureOfNetwork.size()-1, sigmoid);
	NeuralNetwork(structureOfNetwork, activationFunctionByLayer, learningRate, learningRate);
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
	              const std::vector<activationFunctionType>& activationFunctionByLayer,
	              float learningRate,
	              float momentum)
{
	NeuralNetwork(structureOfNetwork, activationFunctionByLayer, learningRate, learningRate);
}
