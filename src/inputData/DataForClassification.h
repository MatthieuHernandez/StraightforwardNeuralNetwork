#pragma once
#include "Data.h"

using namespace data;

class DataForClassification : public Data
{
public:

	DataForClassification();
	int getTrainingLabel(const int index) override;
	int getTestingLabel(const int index) override;
	void loadData() override = 0;
};

