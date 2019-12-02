#include "StraightforwardOption.hpp"

using namespace snn;
using namespace internal;

StraightforwardOption& StraightforwardOption::operator=(const StraightforwardOption& option)
{
	this->autoSaveWhenBetter = option.autoSaveWhenBetter;
	this->saveFilePath = option.saveFilePath;
	this->NeuralNetworkOption::operator=(option);
	return *this;
}

bool StraightforwardOption::operator==(const StraightforwardOption& option) const
{
	return this->NeuralNetworkOption::operator==(option)
		&& this->autoSaveWhenBetter == option.autoSaveWhenBetter
		&& this->saveFilePath == option.saveFilePath;
}
