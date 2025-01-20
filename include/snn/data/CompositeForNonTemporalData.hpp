#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
class CompositeForNonTemporalData : public TemporalComposite
{
    public:
        CompositeForNonTemporalData(snn::Set sets[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool needToTrainOnTrainingData(int index) const override;
        [[nodiscard]] bool needToEvaluateOnTestingData(int index) const override;

        [[nodiscard]] int isValid() override;
};
}  // namespace snn::internal
