#pragma once
#include "Data.h"

class DataForRegression : public Data
{
public:
	std::vector<float>& getTestingOutputs(const int index) override;
};

