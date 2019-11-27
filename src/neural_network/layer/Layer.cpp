#include "layer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

int Layer::isValid() const
{
	if(this->neurons.size() != this->numberOfNeurons
		|| this->numberOfNeurons < 1
		|| this->numberOfNeurons > 1000000)
		return 201;

	for (auto& neuron : this->neurons)
	{
		int err = neuron.isValid();
		if(err != 0)
			return err;
	}
	return 0;
}

Perceptron& Layer::getNeuron(int neuronNumber)
{
	return this->neurons[neuronNumber];
}

Layer& Layer::operator=(const Layer& layer)
{
	this->numberOfInputs = layer.numberOfInputs;
	this->numberOfNeurons = layer.numberOfNeurons;
	this->errors = layer.errors;
	this->neurons = layer.neurons;
	this->learningRate = layer.learningRate;
	this->momentum = layer.momentum;
	return *this;
}

bool Layer::operator==(const Layer& layer) const
{
	return this->numberOfInputs == layer.numberOfInputs
		&& this->numberOfNeurons == layer.numberOfNeurons
		&& this->errors == layer.errors
		&& this->neurons == layer.neurons
		&& this->learningRate == layer.learningRate
		&& this->momentum == layer.momentum;
}

bool Layer::operator!=(const Layer& layer) const
{
	return !this->operator==(layer);
}
