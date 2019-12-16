#pragma once
#include "../Dataset.hpp"

class Iris final : public DataSet
{
public:
	void loadData() override;
};

