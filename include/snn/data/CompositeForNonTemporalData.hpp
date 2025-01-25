#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
class CompositeForNonTemporalData : public TemporalComposite
{
    public:
        CompositeForNonTemporalData(snn::Set sets[2]);

        void shuffle() final;
        void unshuffle() final;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool final;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool final;

        [[nodiscard]] auto isValid() const -> ErrorType final;
};
}  // namespace snn::internal
