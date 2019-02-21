#include "alltoall.h"
#include <omp.h>
#pragma warning(push, 0) 
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)

BOOST_CLASS_EXPORT(AllToAll);

using namespace std;

AllToAll::AllToAll(const int numberOfInputs,
                   const int numberOfNeurons,
                   activationFunctionType function,
                   float learningRate,
                   float momentum)
{
	this->numberOfInputs = numberOfInputs;
	this->numberOfNeurons = numberOfNeurons;
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->neurons.reserve(numberOfNeurons);
	this->outputs.resize(numberOfNeurons);
	this->errors.resize(numberOfInputs);

	for (int n = 0; n < numberOfNeurons; ++n)
	{
		this->neurons.emplace_back(numberOfInputs, function, learningRate, momentum);
	}
}

vector<float>& AllToAll::output(const vector<float>& inputs)
{
	//#pragma omp parallel for TODO : inputs is shared
	for (int n = 0; n < numberOfNeurons; ++n)
	{
		outputs[n] = neurons[n].output(inputs);
	}
	return outputs;
}

vector<float>& AllToAll::backOutput(vector<float>& inputsError)
{

	//#pragma omp parallel for
	for (int n = 0; n < numberOfInputs; ++n)
	{
		errors[n] = 0;
	}

	//#pragma omp parallel for
	for (int n = 0; n < numberOfNeurons; ++n)
	{
		auto result = neurons[n].backOutput(inputsError[n]);
		for (int r = 0; r < numberOfInputs; ++r)
			errors[r] += result[r];
	}
	return errors;
}

void AllToAll::train(vector<float>& inputsError)
{
	for (int n = 0; n < numberOfNeurons; ++n)
	{
		neurons[n].backOutput(inputsError[n]);
	}
}

LayerType AllToAll::getType() const
{
	return allToAll;
}

Layer& AllToAll::equal(const Layer& layer)
{
	return this->Layer::equal(layer);
}

bool AllToAll::operator==(const AllToAll& layer) const
{
	return this->Layer::operator==(layer);
}

bool AllToAll::operator!=(const AllToAll& layer) const
{
	return this->Layer::operator!=(layer);
}