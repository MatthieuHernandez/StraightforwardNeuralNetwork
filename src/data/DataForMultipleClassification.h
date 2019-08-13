#pragma once
#include "Data.h"

namespace snn
{
	class DataForMultipleClassification : public snn::Data
	{
	private:
		int getTrainingLabel(const int index) override;
		int getTestingLabel(const int index) override;
		std::vector<float>& getTestingOutputs(const int index) override;

	public:
		DataForMultipleClassification(std::vector<std::vector<float>>& trainingInputs,
		                              std::vector<std::vector<float>>& trainingLabels,
		                              std::vector<std::vector<float>>& testingInputs,
		                              std::vector<std::vector<float>>& testingLabels,
		                              float separator = 0.5f);

		DataForMultipleClassification(std::vector<std::vector<float>>& inputs,
		                              std::vector<std::vector<float>>& labels,
		                              float separator = 0.5f);
	};
}
