#pragma once
#include "../tools/Error.hpp"
#include "Data.hpp"

namespace snn::internal
{
class TemporalComposite
{
    protected:
        Data* data;

    public:
        TemporalComposite(const TemporalComposite&) = default;
        TemporalComposite(TemporalComposite&&) = delete;
        auto operator=(const TemporalComposite&) -> TemporalComposite& = default;
        auto operator=(TemporalComposite&&) -> TemporalComposite& = delete;
        explicit TemporalComposite(Data* data);
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
