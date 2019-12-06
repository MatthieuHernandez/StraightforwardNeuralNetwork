#include "alltoall.hpp"
#include <omp.h>
#pragma warning(push, 0) 
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(AllToAll);

AllToAll::AllToAll(const int numberOfInputs,
                   const int numberOfNeurons,
                   activationFunctionType function,
                   LayerOption* option)
	: Layer(numberOfInputs, numberOfNeurons, option)
{
	this->neurons.reserve(numberOfNeurons);

	for (int n = 0; n < numberOfNeurons; ++n)
	{
		this->neurons.emplace_back(numberOfInputs, function, this->option);
	}
}

vector<float> AllToAll::output(const vector<float>& inputs)
{
	vector<float> outputs(this->numberOfNeurons);
	for (int n = 0; n < numberOfNeurons; ++n)
	{
		outputs[n] = neurons[n].output(inputs);
	}
	return outputs;
}

vector<float> AllToAll::backOutput(vector<float>& inputsError)
{
	vector<float> errors(this->numberOfInputs);
	for (int n = 0; n < numberOfInputs; ++n)
	{
		errors[n] = 0;
	}

	for (int n = 0; n < numberOfNeurons; ++n)
	{
		auto& result = neurons[n].backOutput(inputsError[n]);
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

int AllToAll::isValid() const
{
	return this->Layer::isValid();
}

LayerType AllToAll::getType() const
{
	return allToAll;
}

Layer& AllToAll::operator=(const Layer& layer)
{
	return this->Layer::operator=(layer);
}

bool AllToAll::operator==(const AllToAll& layer) const
{
	return this->Layer::operator==(layer);
}

bool AllToAll::operator!=(const AllToAll& layer) const
{
	return this->Layer::operator!=(layer);
}