#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForNonTemporalData : TemporalComposite
    {
    public:
        CompositeForNonTemporalData(Set sets[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] int isValid() override;
    };
}
