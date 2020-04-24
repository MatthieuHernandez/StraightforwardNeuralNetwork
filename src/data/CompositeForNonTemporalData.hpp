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

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) override;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) override;

        [[nodiscard]] int isValid() override;
    };
}
