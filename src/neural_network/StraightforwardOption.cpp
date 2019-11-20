#include "StraightforwardOption.hpp"

using namespace snn;

bool StraightforwardOption::operator==(const StraightforwardOption& option) const
{
	return this->autoSaveWhenBetter == option.autoSaveWhenBetter
		&& this->saveFilePath == option.saveFilePath
		&& this->learningRate == option.learningRate
		&& this->momentum == option.momentum;
}
