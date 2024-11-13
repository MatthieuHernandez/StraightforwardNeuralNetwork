#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Tanh final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::tanh; }

    public:
        Tanh()
            : ActivationFunction(-1, 1)
        {
        }

        float function(const float x) const override
        {
            return tanhf(x);
        }

        float derivative(const float x) const override
        {
            return 1 - powf(tanhf(x), 2);
        }
    };
}
