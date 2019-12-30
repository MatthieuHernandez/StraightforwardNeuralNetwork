#include <algorithm>
#include <ctime>
#include <iostream>
#include <boost/serialization/export.hpp>
#include "NeuralNetwork.hpp"
#include "layer/LayerModel.hpp"
#include "../tools/ExtendedExpection.hpp"
#include "layer/LayerFactory.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(NeuralNetwork)

bool NeuralNetwork::isTheFirst = true;

void NeuralNetwork::initialize()
{
	srand(static_cast<int>(time(nullptr)));
	rand();
	ActivationFunction::initialize();
	isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(std::vector<LayerModel>& models)
{
	if (isTheFirst)
		this->initialize();

	option != nullptr ? this->option = option : this->option = new NeuralNetworkOption();

	this->activationFunctionByLayer = activationFunctionByLayer;

	this->layers = LayerFactory::build(models);
	
	this->numberOfLayers = static_cast<int>(this->layers.size()) - 1;
	this->numberOfInput = this->layers[0]->numberOfInputs;
	this->numberOfOutputs = this->layers.back()->numberOfNeurons;
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
	|| this->numberOfLayers > 1000
	|| this->numberOfLayers != this->layers.size()
	|| this->numberOfLayers)
		return 102;

	if (option->learningRate <= 0.0f || option->learningRate >= 1.0f)
		return 103;

	if (option->momentum < 0.0f || option->momentum > 1.0f)
		return 104;

	for (auto& layer : this->layers)
	{
		int err = layer->isValid();
		if(this->option != layer->option)
			return 1001;
		if(err != 0)
			return err;
	}
	return 0;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& neuralNetwork)
{
	this->maxOutputIndex = neuralNetwork.maxOutputIndex;
	this->option = neuralNetwork.option;
	this->numberOfLayers = neuralNetwork.numberOfLayers;
	this->numberOfInput = neuralNetwork.numberOfInput;
	this->numberOfOutputs = neuralNetwork.numberOfOutputs;
	this->activationFunctionByLayer = neuralNetwork.activationFunctionByLayer;
	this->layers.clear();
	this->layers.reserve(neuralNetwork.layers.size());
	this->layers = LayerFactory::copy(neuralNetwork.layers);
	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	return this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& *this->option == *neuralNetwork.option
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& [=] () {
			for (int l = 0; l < numberOfLayers; l++)
			{
				if (*this->layers[l] != *neuralNetwork.layers[l])
					return false;
			}
			return true;
		}();
		//&& equal(this->layers.begin(), this->layers.end(),
		//         neuralNetwork.layers.begin(), neuralNetwork.layers.end());
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
