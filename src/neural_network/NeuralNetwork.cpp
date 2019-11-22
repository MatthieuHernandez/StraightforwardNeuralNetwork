#include <ctime>
//#include <omp.h>
#pragma warning(push, 0)
#include <boost/serialization/vector.hpp>
#pragma warning(pop)
#include "neuralNetwork.hpp"
#include "layer/alltoall.hpp"
#include "../Tools/ExtendedExpection.hpp"
#include <iostream>

using namespace std;
using namespace snn;
using namespace internal;

bool NeuralNetwork::isTheFirst = true;

void NeuralNetwork::initialize()
{
	srand(static_cast<int>(time(nullptr)));
	rand();
	ActivationFunction::initialize();

	//const auto numberOfCore = omp_get_num_procs();
	//omp_set_num_threads(numberOfCore * 2);
	//omp_set_num_threads(128);

	isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structureOfNetwork,
                             const std::vector<activationFunctionType>& activationFunctionByLayer,
                             float learningRate,
                             float momentum) : StatisticAnalysis(structureOfNetwork.back())
{
	if (isTheFirst)
		this->initialize();

	this->structureOfNetwork = structureOfNetwork;
	this->activationFunctionByLayer = activationFunctionByLayer;
	this->learningRate = learningRate;


	this->numberOfLayers = static_cast<int>(structureOfNetwork.size()) - 1;
	this->numberOfHiddenLayers = static_cast<int>(structureOfNetwork.size()) - 2;
	this->numberOfInput = structureOfNetwork[0];
	this->numberOfOutputs = structureOfNetwork.back();

	this->momentum = 0;

	layers.reserve(numberOfLayers);
	for (unsigned int l = 1; l < structureOfNetwork.size(); ++l)
	{
		Layer* layer(new AllToAll(structureOfNetwork[l - 1],
		                          structureOfNetwork[l],
		                          this->activationFunctionByLayer[l - 1],
		                          learningRate,
		                          momentum));
		layers.push_back(layer);
	}

	auto err = this->isValid();
	if (err != 0)
	{
		auto message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of neural networks";
		throw runtime_error(message);
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
	: StatisticAnalysis(neuralNetwork.getNumberOfOutputs())
{
	this->operator=(neuralNetwork);
}

void NeuralNetwork::resetAllNeurons()
{
	// TODO: rework function resetAllNeurons
	throw NotImplementedException();
	/*for(int i = 0; i < neurons.size(); i++)
	{
	    for(int j = 0; j < neurons[i].size(); j++)
	    {
	        neurons[i][j] = Perceptron(neurons[i][j].getNumberOfInputs(), neurons[i][j].getLayerNumber(), neurons[i][j].getNumberInLayer());
	    }
	}*/
}

void NeuralNetwork::addANeuron(int)
{
	// TODO: rework function addANeuron
	throw NotImplementedException();
	/*results[layerNumber].push_back(0);
	errors[layerNumber].push_back(0);
	outputs.push_back(0);

	if(layerNumber == 0)
	{
	    numberOfInput ++;
	    neurons[layerNumber].push_back(Perceptron(neurons[layerNumber][0].getWeights().size(), layerNumber, numberOfInput-1));
	    for(int i = 0; i < neurons[layerNumber+1].size(); i++)
	    {
	        neurons[layerNumber+1][i].addAWeight();
	    }

	}
	else if(layerNumber == numberOfHiddenLayers)
	{
	    numberOfOutputs ++;
	    neurons[layerNumber].push_back(Perceptron(neurons[layerNumber][0].getWeights().size(), layerNumber, numberOfOutputs-1));

	}
	else if(layerNumber > 0 && layerNumber < numberOfHiddenLayers)
	{
	    //numberOfNeuronsInHiddenLayers ++;
	    this->structureOfNetwork[layerNumber+1] ++;
	    for(int j = 1;  j < numberOfHiddenLayers; j++) // output neuron
	    {
	        neurons[j].push_back(Perceptron(neurons[j][0].getWeights().size(), functfqqsdfzsdef);

	        for(int i = 0; i < neurons[j+1].size(); i++)
	        {
	            neurons[j+1][i].addAWeight();
	        }
	    }
	}
	else
	{
	    lastError = 8;
	}*/
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

	if (learningRate < 0 || learningRate > 1)
		return 103;

	if (momentum < 0 || momentum > 1)
		return 104;

	for (auto& layer : this->layers)
	{
		auto err = layer->isValid();
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
	for (const auto& layer : neuralNetwork.layers)
	{
		if (layer->getType() == allToAll)
		{
			auto newLayer = new AllToAll();
			newLayer->operator=(*layer);
			this->layers.push_back(newLayer);
		}
		else
			throw runtime_error("Wrong layer type");
	}

	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	auto equal(this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& this->learningRate == neuralNetwork.learningRate
		&& this->momentum == neuralNetwork.momentum
		&& this->numberOfHiddenLayers == neuralNetwork.numberOfHiddenLayers
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->structureOfNetwork == neuralNetwork.structureOfNetwork
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& this->layers.size() == neuralNetwork.layers.size());

	if (equal)
		for (int l = 0; l < numberOfLayers; l++)
		{
			if (*this->layers[l] != *neuralNetwork.layers[l])
				equal = false;
		}
	return equal;
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
