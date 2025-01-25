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
        CompositeForTimeSeries(snn::Set sets[2], int numberOfRecurrences);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool override;
        [[nodiscard]] auto isFirstTestingDataOfTemporalSequence(int index) const -> bool override;
        [[nodiscard]] auto needToTrainOnTrainingData(int index) const -> bool override;
        [[nodiscard]] auto needToEvaluateOnTestingData(int index) const -> bool override;

        [[nodiscard]] auto isValid() const -> ErrorType override;
};
}  // namespace snn::internal
