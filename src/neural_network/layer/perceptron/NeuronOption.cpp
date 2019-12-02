#include "NeuronOption.hpp"

using namespace snn;

bool NeuronOption::operator==(const NeuronOption& option) const
{
	return this->learningRate == option.learningRate
		&& this->momentum == option.momentum;
}
