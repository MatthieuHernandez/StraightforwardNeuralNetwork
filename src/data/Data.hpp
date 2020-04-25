#pragma once
#include <vector>

#include "ProblemComposite.hpp"
#include "../tools/Tools.hpp"
#include "TemporalComposite.hpp"

namespace snn
{
    enum set
    {
        testing = 0,
        training = 1
    };

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
        continuous,
    };

    struct Set
    {
        int index{0};
        int size{0}; // number of data inside set
        vector2D<float> inputs{};
        vector2D<float> labels{};
        std::vector<int> indexesToShuffle;
        std::vector<bool> areFirstDataOfTemporalSequence{};
        std::vector<bool> needToLearnData{};
    };

    class Data
    {
    private:
        void initialize(problemType problem,
                        std::vector<std::vector<float>>& trainingInputs,
                        std::vector<std::vector<float>>& trainingLabels,
                        std::vector<std::vector<float>>& testingInputs,
                        std::vector<std::vector<float>>& testingLabels,
                        float value,
                        temporalType temporal,
                        int numberOfRecurrence);

        std::unique_ptr<internal::ProblemComposite> problemComposite;
        std::unique_ptr<internal::TemporalComposite> temporalComposite;
        int numberOfRecurrence;

    protected:
        float value;
        void clearData();

        Data(problemType problem,
             std::vector<std::vector<float>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<float>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             float value,
             temporalType temporal,
             int numberOfRecurrence);

        Data(problemType problem,
             std::vector<std::vector<float>>& inputs,
             std::vector<std::vector<float>>& labels,
             float value,
             temporalType temporal,
             int numberOfRecurrence);

        Data(problemType problem,
             std::vector<std::vector<std::vector<float>>>& trainingInputs,
             std::vector<std::vector<float>>& trainingLabels,
             std::vector<std::vector<std::vector<float>>>& testingInputs,
             std::vector<std::vector<float>>& testingLabels,
             float value,
             temporalType temporalType);

        Data(problemType problem,
             std::vector<std::vector<std::vector<float>>>& inputs,
             std::vector<std::vector<float>>& labels,
             float value,
             temporalType temporal);

    public:

        const problemType typeOfProblem;
        const temporalType typeOfTemporal;

        int sizeOfData; // size of one data, equal to size of neural network inputs
        int numberOfLabel; // the number of class, equal to size of neural network outputs

        Set sets[2];

        virtual ~Data() = default;

        void normalization(float min, float max);

        void shuffle();
        void unshuffle();

        [[nodiscard]] int isValid();

        [[nodiscard]] float getValue() const { return value; }

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const;

        [[nodiscard]] const std::vector<float>& getTrainingData(int index) const;
        [[nodiscard]] const std::vector<float>& getTestingData(int index) const;

        [[nodiscard]] int getTrainingLabel(int) const;
        [[nodiscard]] int getTestingLabel(int) const;

        [[nodiscard]] const std::vector<float>& getTrainingOutputs(const int index) const;
        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int) const;

        [[nodiscard]] const std::vector<float>& getData(set set, int index) const;
        [[nodiscard]] const std::vector<float>& getOutputs(set set, int index) const;
        [[nodiscard]] int getLabel(set set, int index) const;
    };
}
