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

NeuralNetwork::NeuralNetwork(int numberOfInputs, vector<LayerModel>& models)
{
	if (isTheFirst)
		this->initialize();

	this->numberOfInput;
	LayerFactory::build(this->layers, this->numberOfInput, models, &this->learningRate, &this->momentum);
	this->numberOfLayers = static_cast<int>(this->layers.size());
	this->numberOfOutputs = this->layers.back()->numberOfNeurons;
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
	: StatisticAnalysis(neuralNetwork.numberOfOutputs)
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

	if (this->learningRate <= 0.0f || this->learningRate >= 1.0f)
		return 103;

	if (this->momentum < 0.0f || this->momentum > 1.0f)
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
	this->numberOfLayers = neuralNetwork.numberOfLayers;
	this->numberOfInput = neuralNetwork.numberOfInput;
	this->numberOfOutputs = neuralNetwork.numberOfOutputs;
	this->layers.clear();
	this->layers.reserve(neuralNetwork.layers.size());
	LayerFactory::copy(this->layers, neuralNetwork.layers);
	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	return this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& this->learningRate == neuralNetwork.learningRate
		&& this->momentum == neuralNetwork.momentum
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& [=] () {
			for (int l = 0; l < numberOfLayers; l++)
			{
				if (*this->layers[l] != *neuralNetwork.layers[l])
					return false;
			}
			return true;
		}();
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
