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

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool override;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool override;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool override;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool override;

        [[nodiscard]] auto isValid() const -> ErrorType override;
};
}  // namespace snn::internal
