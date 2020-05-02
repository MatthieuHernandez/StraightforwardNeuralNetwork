#pragma once
#include "vector"
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForContinuousData : public TemporalComposite
    {
    private:
        int numberOfRecurrence;
        std::vector<int> indexesForShuffles;
        int divide;
        int rest;
    public:
        CompositeForContinuousData(snn::Set sets[2], int numberOfRecurrence);

        void shuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const override;
        [[nodiscard]] bool needToEvaluateOnTestingData(int index) const override;

        [[nodiscard]] int isValid() override;
    };
}
