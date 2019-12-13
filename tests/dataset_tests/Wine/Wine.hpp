#pragma once
#include "../Dataset.hpp"

class Wine final : public Dataset
{
public:
	void loadData() override;
};
