#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
class CompositeForNonTemporalData : public TemporalComposite
{
    public:
        explicit CompositeForNonTemporalData(Dataset* set);

        void shuffle() final;
        void unshuffle() final;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool final;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool final;

        [[nodiscard]] auto isValid() const -> errorType final;
};
}  // namespace snn::internal
