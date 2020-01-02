#include "NeuralNetwork.hpp"

using namespace snn;
using namespace internal;

//=====================================================================
//  Getters
//=====================================================================

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

activationFunction NeuralNetwork::getActivationFunctionInLayer(int layerNumber) const
{
	return this->activationFunctionByLayer[layerNumber];
}

int NeuralNetwork::getNumberOfOutputs() const
{
    return this->numberOfOutputs;
}
