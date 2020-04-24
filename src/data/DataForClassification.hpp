#pragma once
#include "Data.hpp"

namespace snn
{
    class DataForClassification : public Data
    {
    private:
        [[nodiscard]] int getTrainingLabel(const int index) override;

    public:
        DataForClassification(std::vector<std::vector<float>> trainingInputs,
                              std::vector<std::vector<float>> trainingLabels,
                              std::vector<std::vector<float>> testingInputs,
                              std::vector<std::vector<float>> testingLabels,
                              float separator = 0.5f,
                              temporalType type = nonTemporal,
                              int numberOfRecurrence = 0);

        DataForClassification(std::vector<std::vector<float>> trainingInputs,
                              std::vector<std::vector<float>> testingLabels,
                              float separator = 0.5f,
                              temporalType type = nonTemporal,
                              int numberOfRecurrence = 0);

        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int index) override;
        [[nodiscard]] int getTestingLabel(const int index) override;
    };
}
