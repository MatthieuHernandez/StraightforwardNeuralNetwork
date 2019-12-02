#include "StraightforwardOption.hpp"

using namespace snn;

bool StraightforwardOption::operator==(const StraightforwardOption& option) const
{
	return this->NeuralNetworkOption::operator==(option)
		&& this->autoSaveWhenBetter == option.autoSaveWhenBetter
		&& this->saveFilePath == option.saveFilePath;
}
