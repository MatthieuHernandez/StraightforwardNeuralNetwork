#pragma once
#include "../tools/Error.hpp"
#include "Set.hpp"

namespace snn::internal
{
class TemporalComposite
{
    protected:
        Dataset* set;

    public:
        TemporalComposite(const TemporalComposite&) = default;
        TemporalComposite(TemporalComposite&&) = delete;
        auto operator=(const TemporalComposite&) -> TemporalComposite& = default;
        auto operator=(TemporalComposite&&) -> TemporalComposite& = delete;
        explicit TemporalComposite(Dataset* set);
        virtual ~TemporalComposite() = default;

        virtual void shuffle() = 0;
        virtual void unshuffle();

        [[nodiscard]] virtual auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool = 0;
        [[nodiscard]] virtual auto isFirstTestingDataOfTemporalSequence(int index) const -> bool = 0;
        [[nodiscard]] virtual auto needToTrainOnTrainingData(int index) const -> bool = 0;
        [[nodiscard]] virtual auto needToEvaluateOnTestingData(int index) const -> bool = 0;

        [[nodiscard]] virtual auto isValid() const -> errorType;
};
}  // namespace snn::internal
