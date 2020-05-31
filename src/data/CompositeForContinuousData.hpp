#pragma once
#include "vector"
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForContinuousData : public TemporalComposite
    {
    private:
        int numberOfRecurrences;
        std::vector<int> indexesForShuffling;
        int divide;
        int rest;
    public:
        CompositeForContinuousData(snn::Set sets[2], int numberOfRecurrences);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool needToTrainOnTrainingData(int index) const override;
        [[nodiscard]] bool needToEvaluateOnTestingData(int index) const override;

        [[nodiscard]] int isValid() override;
    };
}
