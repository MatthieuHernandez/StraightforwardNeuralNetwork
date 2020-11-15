#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Sigmoid final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::sigmoid; }

    public:
        Sigmoid()
            : ActivationFunction(0, 1)
        {
        }

        float function(const float x) const override
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        float derivative(const float x) const override
        {
            return std::exp(-x) / powf((std::exp(-x) + 1.0f), 2);
        }
    };
}