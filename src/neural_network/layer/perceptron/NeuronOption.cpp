#include <boost/serialization/export.hpp>
#include "NeuronOption.hpp"

using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(NeuronOption)

NeuronOption& NeuronOption::operator=(const NeuronOption& option)
{
	this->learningRate = option.learningRate;
	this->momentum = option.momentum;
	return *this;
}

bool NeuronOption::operator==(const NeuronOption& option) const
{
	return this->learningRate == option.learningRate
		&& this->momentum == option.momentum;
}
