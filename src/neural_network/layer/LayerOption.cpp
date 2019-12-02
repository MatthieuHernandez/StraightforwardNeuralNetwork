#include "LayerOption.hpp"

using namespace snn;
using namespace internal;

LayerOption& LayerOption::operator=(const LayerOption& option)
{
	this->NeuronOption::operator=(option);
	return *this;
}

bool LayerOption::operator==(const LayerOption& option) const
{
	return this->NeuronOption::operator==(option);
}
