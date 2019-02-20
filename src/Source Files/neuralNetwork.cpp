#include "neuralNetwork.h"
#include <ctime>
//#include <omp.h>
#include <fstream>
#pragma warning(push, 0)
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "alltoall.h"
#pragma warning(pop)

using namespace std;

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
	this->lastError = 0;
	this->error = 0;

	errors.resize(numberOfOutputs);
	outputs.resize(numberOfOutputs);

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
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
	: StatisticAnalysis(neuralNetwork.getNumberOfOutputs())
{
	this->operator=(neuralNetwork);
}

void NeuralNetwork::resetAllNeurons()
{
	// TODO: rework function resetAllNeurons
	throw notImplementedException();
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
	throw notImplementedException();
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

void NeuralNetwork::saveAs(std::string filePath)
{
	ofstream file(filePath);
	boost::archive::text_oarchive binaryFile(file);
	binaryFile << this;
	std::ofstream ofs("filename");
	boost::archive::text_oarchive oa(ofs);
}

NeuralNetwork& NeuralNetwork::loadFrom(std::string filePath)
{
	NeuralNetwork* neuralNetwork{};
	ifstream file(filePath, ios::binary);
	boost::archive::text_iarchive binaryFile(file);
	binaryFile >> neuralNetwork;
	return *neuralNetwork;
}

template <class Archive>
void NeuralNetwork::serialize(Archive& ar, const unsigned int version)
{
	boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
	ar & boost::serialization::base_object<StatisticAnalysis>(*this);
	ar & this->maxOutputIndex;
	ar & this->lastError;
	ar & this->learningRate;
	ar & this->error;
	ar & this->momentum;
	ar & this->numberOfHiddenLayers;
	ar & this->numberOfLayers;
	ar & this->numberOfInput;
	ar & this->numberOfOutputs;
	ar & this->structureOfNetwork;
	ar & this->activationFunctionByLayer;
	ar & this->errors;
	ar & this->outputs;
	ar & this->numberOfInput;

	ar.template register_type<AllToAll>();
	ar & layers;
}

int NeuralNetwork::isValid()
{
	//TODO: rework isValid
	throw notImplementedException();
	/*int numberOfWeightsReal = 0;
	int computedNumberWeights = 0;

	for(int i = 0; i < neurons.size(); i++)
	{
	    if(((int)neurons[i].size() !=  structureOfNetwork[i+1] && i < numberOfHiddenLayers)
	    || ((int)neurons[i].size() != numberOfOutputs && i == numberOfHiddenLayers))
	    {
	        lastError = 10;
	        return lastError;
	    }
	}
	for(int i = 0; i < neurons.size(); i++)
	{
	    for(int j = 0; j < neurons[i].size(); j++)
	    {
	       numberOfWeightsReal += neurons[i][j].getNumberOfInputs();
	    }
	}

	for(int i=1; i<structureOfNetwork.size();i++)
	{
	    computedNumberWeights += structureOfNetwork[i-1] * structureOfNetwork[i];
	}

	if(numberOfWeightsReal != computedNumberWeights)
	{
	    lastError = 11;
	    cout << numberOfWeightsReal << endl;
	    cout << computedNumberWeights << endl;
	    return lastError;
	}
	if(numberOfInput < 1 || numberOfInput > 2073600) // 1920 * 1080
	{
	    lastError = 1;
	    return lastError;
	}
	if(numberOfHiddenLayers < 1 || numberOfHiddenLayers > 100)
	{
	    lastError = 2;
	    cout << numberOfHiddenLayers << endl;
	    return lastError;
	}
	if(neurons.size() != (unsigned)(numberOfHiddenLayers+1))
	{
	   lastError = 3;
	   return lastError;
	}
	if(learningRate < 0 || learningRate > 1 )
	{
	    return 5;
	    cout << learningRate << endl;
	}
	for(int i = 0; i < neurons.size(); i++)
	{
	    for(int j = 0; j < neurons[i].size(); j++)
	    {
	        if(neurons[i][j].isValid() != 0)
	        {
	          lastError = neurons[i][j].isValid();
	          return lastError;
	        }
	    }
	}
	return 0;*/
}

int NeuralNetwork::getLastError() const
{
	return lastError;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& neuralNetwork)
{
	this->maxOutputIndex = neuralNetwork.maxOutputIndex;
	this->lastError = neuralNetwork.lastError;
	this->learningRate = neuralNetwork.learningRate;
	this->error = neuralNetwork.error;
	this->momentum = neuralNetwork.momentum;
	this->numberOfHiddenLayers = neuralNetwork.numberOfHiddenLayers;
	this->numberOfLayers = neuralNetwork.numberOfLayers;
	this->numberOfInput = neuralNetwork.numberOfInput;
	this->numberOfOutputs = neuralNetwork.numberOfOutputs;
	this->structureOfNetwork = neuralNetwork.structureOfNetwork;
	this->activationFunctionByLayer = neuralNetwork.activationFunctionByLayer;
	this->errors = neuralNetwork.errors;
	this->outputs = neuralNetwork.outputs;

	this->layers.clear();
	this->layers.reserve(neuralNetwork.layers.size());
	for (const auto& layer : neuralNetwork.layers)
	{
		if (layer->getType() == allToAll)
		{
			AllToAll* newlayer = new AllToAll();
			newlayer->equal(*layer);
			this->layers.push_back(newlayer);
		}
		else
			throw exception();
	}

	return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
	bool equal(this->maxOutputIndex == neuralNetwork.maxOutputIndex
		&& this->lastError == neuralNetwork.lastError
		&& this->learningRate == neuralNetwork.learningRate
		&& this->error == neuralNetwork.error
		&& this->momentum == neuralNetwork.momentum
		&& this->numberOfHiddenLayers == neuralNetwork.numberOfHiddenLayers
		&& this->numberOfLayers == neuralNetwork.numberOfLayers
		&& this->numberOfInput == neuralNetwork.numberOfInput
		&& this->numberOfOutputs == neuralNetwork.numberOfOutputs
		&& this->structureOfNetwork == neuralNetwork.structureOfNetwork
		&& this->activationFunctionByLayer == neuralNetwork.activationFunctionByLayer
		&& this->layers.size() == neuralNetwork.layers.size()
		&& this->errors == neuralNetwork.errors
		&& this->outputs == neuralNetwork.outputs);

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
