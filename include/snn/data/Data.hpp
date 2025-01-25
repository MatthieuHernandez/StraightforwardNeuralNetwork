#pragma once
#include <memory>
#include <vector>

#include "../tools/Error.hpp"
#include "ProblemComposite.hpp"
#include "Set.hpp"
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

class Data
{
    private:
        void initialize(std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
                        std::vector<std::vector<float>>& testingLabels);

        void flatten(set set, std::vector<std::vector<std::vector<float>>>& input3D);
        void flatten(std::vector<std::vector<std::vector<float>>>& input3D);

        std::unique_ptr<internal::ProblemComposite> problemComposite;
        std::unique_ptr<internal::TemporalComposite> temporalComposite;
        int numberOfRecurrences;
        float precision;
        float separator;

        std::vector<float> batchedData{};

    public:
        Data(problem typeOfProblem, std::vector<std::vector<float>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal = nature::nonTemporal,
             int numberOfRecurrences = 0);

        Data(problem typeOfProblem, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels,
             nature temporal = nature::nonTemporal, int numberOfRecurrences = 0);

        Data(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<std::vector<float>>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences = 0);

        Data(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& inputs,
             std::vector<std::vector<float>>& labels, nature typeOfTemporal, int numberOfRecurrences = 0);

        const problem typeOfProblem;
        const nature typeOfTemporal;

        int sizeOfData{};      // size of one data, equal to size of neural network inputs
        int numberOfLabels{};  // the number of class, equal to size of neural network outputs

        Set sets[2];

        virtual ~Data() = default;

        void normalize(float min, float max);

        void shuffle();
        void unshuffle();

        [[nodiscard]] auto isValid() const -> ErrorType;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool;
        [[nodiscard]] auto needToLearnOnTrainingData(int index) const -> bool;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool;

        [[nodiscard]] auto getTrainingData(int index, int batchSize = 1) -> const std::vector<float>&;
        [[nodiscard]] auto getTestingData(int index) const -> const std::vector<float>&;

        [[nodiscard]] auto getTrainingLabel(int index) const -> int;
        [[nodiscard]] auto getTestingLabel(int index) const -> int;

        [[nodiscard]] auto getTrainingOutputs(int index, int batchSize = 1) -> const std::vector<float>&;
        [[nodiscard]] auto getTestingOutputs(int) const -> const std::vector<float>&;

        [[nodiscard]] auto getData(set set, int index) -> const std::vector<float>&;
        [[nodiscard]] auto getOutputs(set set, int index) -> const std::vector<float>&;
        [[nodiscard]] auto getLabel(set set, int index) const -> int;

        [[nodiscard]] auto getSeparator() const -> float;
        void setSeparator(float value);
        [[nodiscard]] auto getPrecision() const -> float;
        void setPrecision(float value);
};
}  // namespace snn
