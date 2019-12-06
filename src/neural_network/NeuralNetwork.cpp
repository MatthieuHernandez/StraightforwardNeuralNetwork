#include <algorithm>
#include <ctime>
#include <iostream>
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
	isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structureOfNetwork,
                             const std::vector<activationFunctionType>& activationFunctionByLayer,
                             NeuralNetworkOption* option) : StatisticAnalysis(structureOfNetwork.back())
{
	if (isTheFirst)
		this->initialize();

	option != nullptr ? this->option = option : this->option = new NeuralNetworkOption();

	this->structureOfNetwork = structureOfNetwork;
	this->activationFunctionByLayer = activationFunctionByLayer;

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
		                          this->option));
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

	if (option->learningRate <= 0.0f || option->learningRate >= 1.0f)
		return 103;

	if (option->momentum < 0.0f || option->momentum > 1.0f)
		return 104;

	for (auto& layer : this->layers)
	{
		int err = layer->isValid();
		if(this->option != layer->option)
			return 10019;
		if(err != 0)
			return err;
	}
	return 0;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& neuralNetwork)
{
	this->maxOutputIndex = neuralNetwork.maxOutputIndex;
	this->option = neuralNetwork.option;
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
	return this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& this->option == neuralNetwork.option
		&& this->numberOfHiddenLayers == neuralNetwork.numberOfHiddenLayers
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->structureOfNetwork == neuralNetwork.structureOfNetwork
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& equal(this->layers.begin(), this->layers.end(),
		         neuralNetwork.layers.begin(), neuralNetwork.layers.end());
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
