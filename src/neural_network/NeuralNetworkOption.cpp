#include <boost/serialization/export.hpp>
#include "NeuralNetworkOption.hpp"

using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(NeuralNetworkOption)

NeuralNetworkOption& NeuralNetworkOption::operator=(const NeuralNetworkOption& option)
{
	this->LayerOption::operator=(option);
	return *this;
}

bool NeuralNetworkOption::operator==(const NeuralNetworkOption& option) const
{
	return this->LayerOption::operator==(option);
}
