#pragma once
#include "Data.h"

class DataForClassification : public Data
{
public:
	int getTrainingLabel(const int index) override;
	int getTestingLabel(const int index) override;
	void loadData() override = 0;
};

