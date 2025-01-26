#pragma once
#include <vector>

#include "TemporalComposite.hpp"

namespace snn::internal
{
class CompositeForTimeSeries : public TemporalComposite
{
    private:
        int numberOfRecurrences;
        std::vector<int> indexesForShuffling;
        int offset;
        int divide;
        int rest;

    public:
        CompositeForTimeSeries(Dataset* set, int numberOfRecurrences);

        void shuffle() final;
        void unshuffle() final;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool final;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool final;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool final;

        [[nodiscard]] auto isValid() const -> errorType final;
};
}  // namespace snn::internal
