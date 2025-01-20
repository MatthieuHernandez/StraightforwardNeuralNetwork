#pragma once
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

        [[nodiscard]] virtual bool isFirstTrainingDataOfTemporalSequence(int index) const = 0;
        [[nodiscard]] virtual bool isFirstTestingDataOfTemporalSequence(int index) const = 0;
        [[nodiscard]] virtual bool needToTrainOnTrainingData(int index) const = 0;
        [[nodiscard]] virtual bool needToEvaluateOnTestingData(int index) const = 0;

        [[nodiscard]] virtual int isValid();
};
}  // namespace snn::internal
