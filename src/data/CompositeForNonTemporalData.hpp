#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForNonTemporalData : TemporalComposite
    {
    public:
        CompositeForNonTemporalData(Set sets[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index)  const override;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const override;

        [[nodiscard]] int isValid() override;
    };
}
