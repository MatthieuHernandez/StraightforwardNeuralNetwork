#pragma once
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Identity : public ActivationFunction
    {
    private:
        activationFunction getType() const override { return identity; }

    public:
        float function(const float x) const override
        {
            return x;
        }

        float derivative(const float x) const override
        {
            return 1.0f;
        }
    };
}