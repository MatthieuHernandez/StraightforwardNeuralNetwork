#pragma once
#include "../tools/Error.hpp"
#include "Set.hpp"

namespace snn::internal
{
class TemporalComposite
{
    protected:
        Set* sets;

    public:
        TemporalComposite(snn::Set sets[2]);
        virtual ~TemporalComposite() = default;

        virtual void shuffle() = 0;
        virtual void unshuffle();

        [[nodiscard]] virtual auto isFirstTrainingDataOfTemporalSequence(int index) const -> bool = 0;
        [[nodiscard]] virtual auto isFirstTestingDataOfTemporalSequence(int index) const -> bool = 0;
        [[nodiscard]] virtual auto needToTrainOnTrainingData(int index) const -> bool = 0;
        [[nodiscard]] virtual auto needToEvaluateOnTestingData(int index) const -> bool = 0;

        [[nodiscard]] virtual auto isValid() const -> ErrorType;
};
}  // namespace snn::internal
