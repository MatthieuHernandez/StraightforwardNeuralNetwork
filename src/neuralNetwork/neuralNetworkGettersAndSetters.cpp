#include "neuralNetwork.h"

//=====================================================================
//  Getters and setters
//=====================================================================

void NeuralNetwork::setLearningRate(const float learningRate)
{
    this->learningRate = learningRate;
}

float NeuralNetwork::getLearningRate() const
{
    return learningRate;
}

void NeuralNetwork::setMomentum(const float value)
{
    if(value >= 0.0f && value <= 1.0f)
    {
        this->momentum = value;
    }
    else
    {
		lastError = 16;
    }
}

float NeuralNetwork::getMomentum() const
{
    return this->momentum;
}

//=====================================================================
//  Only getters
//=====================================================================

Layer* NeuralNetwork::getLayer(const int layerNumber)
{
    return this->layers[layerNumber];
}

int NeuralNetwork::getNumberOfInputs() const
{
    return this->numberOfInput;
}

int NeuralNetwork::getNumberOfHiddenLayers() const
{
    return this->numberOfHiddenLayers;
}

int NeuralNetwork::getNumberOfNeuronsInLayer(const int layerNumber) const
{
    return this->structureOfNetwork[layerNumber+1];
}

activationFunctionType NeuralNetwork::getActivationFunctionInLayer(int layerNumber) const
{
	return this->activationFunctionByLayer[layerNumber];
}

int NeuralNetwork::getNumberOfOutputs() const
{
    return this->numberOfOutputs;
}
