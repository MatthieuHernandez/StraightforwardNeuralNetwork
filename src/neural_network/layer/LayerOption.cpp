#include "LayerOption.hpp"
#include <boost/serialization/export.hpp>

using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LayerOption)

LayerOption& LayerOption::operator=(const LayerOption& option)
{
	this->NeuronOption::operator=(option);
	return *this;
}

bool LayerOption::operator==(const LayerOption& option) const
{
	return this->NeuronOption::operator==(option);
}
