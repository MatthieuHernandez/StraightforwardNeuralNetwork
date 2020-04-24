#pragma once
#include "Data.hpp"

namespace snn::internal
{
    class TemporalComposite
    {
    protected:
        Set sets[2];

    public:
        TemporalComposite(Set set[2]);

        virtual void shuffle() = 0;
        virtual void unshuffle() = 0;

        [[nodiscard]] virtual int isValid();
    };
}
