#pragma once
#include "Data.h"

using namespace data;

class DataForRegression : public Data
{
public:

	DataForRegression();
	std::vector<float>& getTestingOutputs(const int index) override;
	void loadData() override = 0;

	//std::vector<float>& getTrainingData(const int index);
	//std::vector<float>& getTestingData(const int index);
};

