#include "NeuralNetworkOption.hpp"

using namespace snn;

bool NeuralNetworkOption::operator==(const NeuralNetworkOption& option) const
{
	return this->LayerOption::operator==(option);
}
