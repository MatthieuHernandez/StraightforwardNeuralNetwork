#include "StraightforwardOption.h"

using namespace snn;

bool StraightforwardOption::operator==(const StraightforwardOption& option) const
{
	return this->autoSaveWhenBetter == option.autoSaveWhenBetter;
}
