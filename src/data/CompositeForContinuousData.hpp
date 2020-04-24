#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForContinuousData : TemporalComposite
    {
    public:
        CompositeForContinuousData(Set set[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const override;

        [[nodiscard]] int isValid() override;

    };
}
