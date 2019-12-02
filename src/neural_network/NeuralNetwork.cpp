#include <algorithm>
#include <ctime>
#include <iostream>
//#pragma warning(push, 0)
#include <boost/serialization/vector.hpp>
//#pragma warning(pop)
#include "NeuralNetwork.hpp"
#include "layer/AllToAll.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

bool NeuralNetwork::isTheFirst = true;

void NeuralNetwork::initialize()
{
	srand(static_cast<int>(time(nullptr)));
	rand();
	ActivationFunction::initialize();

	//int auto numberOfCore = omp_get_num_procs();
	//omp_set_num_threads(numberOfCore * 2);
	//omp_set_num_threads(128);

	isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structureOfNetwork,
                             const std::vector<activationFunctionType>& activationFunctionByLayer,
                             float& learningRate,
                             float& momentum) : StatisticAnalysis(structureOfNetwork.back())
{
	if (isTheFirst)
		this->initialize();

	this->structureOfNetwork = structureOfNetwork;
	this->activationFunctionByLayer = activationFunctionByLayer;

	this->learningRate = &learningRate;
	this->momentum = &momentum;

	this->numberOfLayers = static_cast<int>(structureOfNetwork.size()) - 1;
	this->numberOfHiddenLayers = static_cast<int>(structureOfNetwork.size()) - 2;
	this->numberOfInput = structureOfNetwork[0];
	this->numberOfOutputs = structureOfNetwork.back();


	layers.reserve(numberOfLayers);
	for (unsigned int l = 1; l < structureOfNetwork.size(); ++l)
	{
		layers.emplace_back(new AllToAll(structureOfNetwork[l - 1],
		                          structureOfNetwork[l],
		                          this->activationFunctionByLayer[l - 1],
		                          learningRate,
		                          momentum));
	}

	int err = this->isValid();
	if (err != 0)
	{
		string message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of neural networks";
		throw runtime_error(message);
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
	: StatisticAnalysis(neuralNetwork.getNumberOfOutputs())
{
	this->operator=(neuralNetwork);
}

int NeuralNetwork::isValid() const
{
	//TODO: rework isValid
	if (this->numberOfInput < 1 
	|| this->numberOfInput > 2073600) // 1920 * 1080
		return 101;

	if (this->numberOfLayers < 1 
	|| this->numberOfHiddenLayers > 1000
	|| this->numberOfLayers != this->layers.size()
	|| this->numberOfLayers != numberOfHiddenLayers + 1)
		return 102;

	if (*learningRate <= 0.0f || *learningRate >= 1.0f)
		return 103;

	if (*momentum < 0.0f || *momentum > 1.0f)
		return 104;

	for (auto& layer : this->layers)
	{
		int err = layer->isValid();
		if(err != 0)
			return err;
	}
	return 0;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& neuralNetwork)
{
	this->maxOutputIndex = neuralNetwork.maxOutputIndex;
	this->learningRate = neuralNetwork.learningRate;
	this->momentum = neuralNetwork.momentum;
	this->numberOfHiddenLayers = neuralNetwork.numberOfHiddenLayers;
	this->numberOfLayers = neuralNetwork.numberOfLayers;
	this->numberOfInput = neuralNetwork.numberOfInput;
	this->numberOfOutputs = neuralNetwork.numberOfOutputs;
	this->structureOfNetwork = neuralNetwork.structureOfNetwork;
	this->activationFunctionByLayer = neuralNetwork.activationFunctionByLayer;

	this->layers.clear();
	this->layers.reserve(neuralNetwork.layers.size());
	for (auto& layer : neuralNetwork.layers)
	{
		if (layer->getType() == allToAll)
		{
			auto newLayer = new AllToAll();
			newLayer->operator=(*layer);
			this->layers.emplace_back(newLayer);
		}
		else
			throw runtime_error("Wrong layer type");
	}

	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	bool isEqual = this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& *this->learningRate == *neuralNetwork.learningRate
		&& *this->momentum == *neuralNetwork.momentum
		&& this->numberOfHiddenLayers == neuralNetwork.numberOfHiddenLayers
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->structureOfNetwork == neuralNetwork.structureOfNetwork
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& equal(this->layers.begin(), this->layers.end(),
		         neuralNetwork.layers.begin(), neuralNetwork.layers.end());
	return isEqual;
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
