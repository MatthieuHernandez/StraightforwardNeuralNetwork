#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
class CompositeForTemporalData : public TemporalComposite
{
    private:
        std::vector<int> indexesForShuffles;

    public:
        CompositeForTemporalData(snn::Set sets[2]);

        void shuffle() final;
        void unshuffle() final;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool final;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool final;

        [[nodiscard]] auto isValid() const -> ErrorType final;
};
}  // namespace snn::internal
