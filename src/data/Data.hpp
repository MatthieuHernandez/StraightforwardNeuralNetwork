#pragma once
#include <memory>
#include <vector>
#include "Set.hpp"
#include "ProblemComposite.hpp"
#include "TemporalComposite.hpp"

namespace snn
{
    enum class problem
    {
        classification,
        multipleClassification,
        regression
    };

    enum class nature
    {
        nonTemporal,
        sequential,
        timeSeries,
    };

    class Data
    {
    private:
        void initialize(std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels,
                        std::vector<std::vector<float>>& testingInputs,
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
        Data(problem typeOfProblem,
             std::vector<std::vector<float>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<float>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             nature typeOfTemporal = nature::nonTemporal,
             int numberOfRecurrences = 0);

        Data(problem typeOfProblem,
             std::vector<std::vector<float>>& inputs,
             std::vector<std::vector<float>>& labels,
             nature temporal = nature::nonTemporal,
             int numberOfRecurrences = 0);

        Data(problem typeOfProblem,
             std::vector<std::vector<std::vector<float>>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<std::vector<float>>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             nature typeOfTemporal,
             int numberOfRecurrences = 0);

        Data(problem typeOfProblem,
             std::vector<std::vector<std::vector<float>>>& inputs,
             std::vector<std::vector<float>>& labels,
             nature typeOfTemporal,
             int numberOfRecurrences = 0);

        const problem typeOfProblem;
        const nature typeOfTemporal;

        int sizeOfData{}; // size of one data, equal to size of neural network inputs
        int numberOfLabels{}; // the number of class, equal to size of neural network outputs

        Set sets[2];

        virtual ~Data() = default;

        void normalization(float min, float max);

        void shuffle();
        void unshuffle();

        [[nodiscard]] int isValid();

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const;
        [[nodiscard]] bool needToEvaluateOnTestingData(int index) const;

        [[nodiscard]] const std::vector<float>& getTrainingData(int index, int batchSize = 1);
        [[nodiscard]] const std::vector<float>& getTestingData(int index) const;

        [[nodiscard]] int getTrainingLabel(int) const;
        [[nodiscard]] int getTestingLabel(int) const;

        [[nodiscard]] const std::vector<float>& getTrainingOutputs(int index, int batchSize = 1);
        [[nodiscard]] const std::vector<float>& getTestingOutputs(int) const;

        [[nodiscard]] const std::vector<float>& getData(set set, int index);
        [[nodiscard]] const std::vector<float>& getOutputs(set set, int index);
        [[nodiscard]] int getLabel(set set, int index) const;

        [[nodiscard]] float getSeparator() const;
        void setSeparator(float value);
        [[nodiscard]] float getPrecision() const;
        void setPrecision(float value);
    };
}
