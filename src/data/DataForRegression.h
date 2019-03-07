#pragma once
#include "Data.h"

class DataForRegression : public snn::Data
{
private:
	std::vector<float>& getTestingOutputs(const int index) override;

public:
	DataForRegression(std::vector<std::vector<float>>& trainingInputs,
	                  std::vector<std::vector<float>>& trainingLabels,
	                  std::vector<std::vector<float>>& testingInputs,
	                  std::vector<std::vector<float>>& testingLabels);
};
