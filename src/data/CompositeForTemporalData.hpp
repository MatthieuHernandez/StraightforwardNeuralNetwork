#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForTemporalData : TemporalComposite
    {
    public:
        CompositeForTemporalData(Set set[2]);

        void shuffle() override;
        void unshuffle() override;

        [[nodiscard]] int isValid() override;
    };
}
