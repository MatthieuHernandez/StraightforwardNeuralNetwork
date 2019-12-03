#include "layer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Layer::Layer(int numberOfInputs,
             int numberOfNeurons,
             LayerOption* option)
{
	this->numberOfInputs = numberOfInputs;
	this->numberOfNeurons = numberOfNeurons;
	this->option = option;
}

int Layer::isValid() const
{
	if(this->neurons.size() != this->numberOfNeurons
		|| this->numberOfNeurons < 1
		|| this->numberOfNeurons > 1000000)
		return 201;

	for (auto& neuron : this->neurons)
	{
		int err = neuron.isValid();
		if(this->option != neuron.option)
			return 1001;
		if(err != 0)
			return err;
	}
	return 0;
}

Layer& Layer::operator=(const Layer& layer)
{
	this->option = layer.option;
	this->numberOfInputs = layer.numberOfInputs;
	this->numberOfNeurons = layer.numberOfNeurons;
	this->errors = layer.errors;
	this->neurons = layer.neurons;
	return *this;
}

bool Layer::operator==(const Layer& layer) const
{
	return this->option == layer.option
		&& this->numberOfInputs == layer.numberOfInputs
		&& this->numberOfNeurons == layer.numberOfNeurons
		&& this->errors == layer.errors
		&& this->neurons == layer.neurons;
}

bool Layer::operator!=(const Layer& layer) const
{
	return !this->operator==(layer);
}
