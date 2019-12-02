#include "LayerOption.hpp"

using namespace snn;

bool LayerOption::operator==(const LayerOption& option) const
{
	return this->NeuronOption::operator==(option);
}
