#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForContinuousData : TemporalComposite
    {
    public:
        CompositeForContinuousData(Set set[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] int isValid() override;
    };
}
