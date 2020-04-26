#pragma once
#include "Set.hpp"

namespace snn::internal
{
    class TemporalComposite
    {
    protected:
        snn::Set sets[2];

    public:
        TemporalComposite(snn::Set sets[2]);

        virtual void shuffle() = 0;
        virtual void unshuffle() = 0;

        [[nodiscard]] virtual bool isFirstTrainingDataOfTemporalSequence(int index) const = 0;
        [[nodiscard]] virtual bool isFirstTestingDataOfTemporalSequence(int index) const = 0;
        [[nodiscard]] virtual bool needToLearnOnTrainingData(int index) const = 0;
        [[nodiscard]] virtual bool needToEvaluateOnTestingData(int index) const = 0;

        [[nodiscard]] virtual int isValid();
    };
}