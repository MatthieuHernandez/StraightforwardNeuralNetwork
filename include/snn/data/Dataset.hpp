#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "../tools/Error.hpp"
#include "Data.hpp"
#include "ProblemComposite.hpp"
#include "TemporalComposite.hpp"

namespace snn
{
enum class problem : uint8_t
{
    classification,
    multipleClassification,
    regression
};

enum class nature : uint8_t
{
    nonTemporal,
    sequential,
    timeSeries,
};

const float defaultPrecision = 0.1F;
const float defaultSeparator = 0.5F;

class Dataset
{
    private:
        void initialize(std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
                        std::vector<std::vector<float>>& testingLabels);

        void flatten(internal::Set& set, std::vector<std::vector<std::vector<float>>>& input3D);
        void flatten(std::vector<std::vector<std::vector<float>>>& input3D);

        std::unique_ptr<internal::ProblemComposite> problemComposite;
        std::unique_ptr<internal::TemporalComposite> temporalComposite;
        int numberOfRecurrences{};
        float precision = defaultPrecision;
        float separator = defaultSeparator;

        std::vector<float> batchedData;

    public:
        Dataset(problem typeOfProblem, std::vector<std::vector<float>>& trainingInputs,
                std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
                std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal = nature::nonTemporal,
                int numberOfRecurrences = 0);
        Dataset(const Dataset&) = delete;
        Dataset(Dataset&&) = delete;
        auto operator=(const Dataset&) -> Dataset& = delete;
        auto operator=(Dataset&&) -> Dataset& = delete;

        Dataset(problem typeOfProblem, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels,
                nature temporal = nature::nonTemporal, int numberOfRecurrences = 0);

        Dataset(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& trainingInputs,
                std::vector<std::vector<float>>& trainingLabels,
                std::vector<std::vector<std::vector<float>>>& testingInputs,
                std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences = 0);

        Dataset(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& inputs,
                std::vector<std::vector<float>>& labels, nature typeOfTemporal, int numberOfRecurrences = 0);

        const problem typeOfProblem;
        const nature typeOfTemporal;

        int sizeOfData{};      // size of one data, equal to size of neural network inputs
        int numberOfLabels{};  // the number of class, equal to size of neural network outputs

        internal::Data data;

        virtual ~Dataset() = default;

        void normalize(float min, float max);

        void shuffle();
        void unshuffle();

        [[nodiscard]] auto isValid() const -> errorType;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool;
        [[nodiscard]] auto needToLearnOnTrainingData(int index) const -> bool;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool;

        [[nodiscard]] auto getTrainingData(int index, int batchSize = 1) -> const std::vector<float>&;
        [[nodiscard]] auto getTestingData(int index) const -> const std::vector<float>&;

        [[nodiscard]] auto getTrainingLabel(int index) const -> int;
        [[nodiscard]] auto getTestingLabel(int index) const -> int;

        [[nodiscard]] auto getTrainingOutputs(int index, int batchSize = 1) -> const std::vector<float>&;
        [[nodiscard]] auto getTestingOutputs(int index) const -> const std::vector<float>&;

        [[nodiscard]] auto getSeparator() const -> float;
        void setSeparator(float value);
        [[nodiscard]] auto getPrecision() const -> float;
        void setPrecision(float value);
};
}  // namespace snn
