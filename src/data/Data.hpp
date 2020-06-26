#pragma once
#include <memory>
#include <vector>
#include "Set.hpp"
#include "ProblemComposite.hpp"
#include "TemporalComposite.hpp"

namespace snn
{
    enum problemType
    {
        classification,
        multipleClassification,
        regression
    };

    enum temporalType
    {
        nonTemporal,
        temporal,
        timeSeries,
    };

    class Data
    {
    private:
        void initialize(problemType typeOfProblem,
                        std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels,
                        std::vector<std::vector<float>>& testingInputs,
                        std::vector<std::vector<float>>& testingLabels,
                        temporalType typeOfTemporal,
                        int numberOfRecurrences);

        void flatten(set set, std::vector<std::vector<std::vector<float>>>& input3D);

        std::unique_ptr<internal::ProblemComposite> problemComposite;
        std::unique_ptr<internal::TemporalComposite> temporalComposite;
        int numberOfRecurrences{};
        float precision{};
        float separator{};

    public:
        Data(problemType typeOfProblem,
             std::vector<std::vector<float>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<float>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             temporalType typeOfTemporal = nonTemporal,
             int numberOfRecurrences = 0);

        Data(problemType typeOfProblem,
             std::vector<std::vector<float>>& inputs,
             std::vector<std::vector<float>>& labels,
             temporalType temporal = nonTemporal,
             int numberOfRecurrences = 0);

        Data(problemType typeOfProblem,
             std::vector<std::vector<std::vector<float>>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<std::vector<float>>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             temporalType typeOfTemporal,
             int numberOfRecurrences = 0);

        Data(problemType typeOfProblem,
             std::vector<std::vector<std::vector<float>>>& inputs,
             std::vector<std::vector<float>>& labels,
             temporalType typeOfTemporal,
             int numberOfRecurrences = 0);

        const problemType typeOfProblem;
        const temporalType typeOfTemporal;

        int sizeOfData{}; // size of one data, equal to size of neural network inputs
        int numberOfLabel{}; // the number of class, equal to size of neural network outputs

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

        [[nodiscard]] const std::vector<float>& getTrainingData(int index) const;
        [[nodiscard]] const std::vector<float>& getTestingData(int index) const;

        [[nodiscard]] int getTrainingLabel(int) const;
        [[nodiscard]] int getTestingLabel(int) const;

        [[nodiscard]] const std::vector<float>& getTrainingOutputs(const int index) const;
        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int) const;

        [[nodiscard]] const std::vector<float>& getData(set set, int index) const;
        [[nodiscard]] const std::vector<float>& getOutputs(set set, int index) const;
        [[nodiscard]] int getLabel(set set, int index) const;

        
        [[nodiscard]] float getSeparator() const;
        void setSeparator(float value);
        [[nodiscard]] float getPrecision() const;
        void setPrecision(float value);


    };
}
