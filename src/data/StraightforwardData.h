#pragma once
#include "Data.h"
#include <vector>

namespace snn
{
	class StraightforwardData
	{
	public:
		StraightforwardData(problemType type,
		                    std::vector<std::vector<float>>& Inputs,
		                    std::vector<std::vector<float>>& Labels);

		StraightforwardData(problemType type,
		                    std::vector<std::vector<float>>& trainingInputs,
		                    std::vector<std::vector<float>>& trainingLabels,
		                    std::vector<std::vector<float>>& testingInputs,
		                    std::vector<std::vector<float>>& testingLabels);
	private:
		Data* data;
		static std::vector<std::vector<float>> NULL_VECTOR;
	};
}
