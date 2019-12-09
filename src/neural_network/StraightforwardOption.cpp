#include "StraightforwardOption.hpp"
#include <boost/serialization/export.hpp>

using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(StraightforwardOption)

StraightforwardOption& StraightforwardOption::operator=(const StraightforwardOption& option)
{
	this->autoSaveWhenBetter = option.autoSaveWhenBetter;
	this->autoSaveFilePath = option.autoSaveFilePath;
	this->NeuralNetworkOption::operator=(option);
	return *this;
}

bool StraightforwardOption::operator==(const StraightforwardOption& option) const
{
	return this->NeuralNetworkOption::operator==(option)
		&& this->autoSaveWhenBetter == option.autoSaveWhenBetter
		&& this->autoSaveFilePath == option.autoSaveFilePath;
}
