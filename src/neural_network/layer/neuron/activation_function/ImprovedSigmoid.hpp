#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class ImprovedSigmoid final : public ActivationFunction
    {
    private:
        activationFunction getType() const override { return iSigmoid; }

    public:
        ImprovedSigmoid()
            : ActivationFunction(-INFINITY, +INFINITY)
        {
        }

        float function(const float x) const override
        {
            return 1.0f / (1.0f + std::exp(-x)) + x * 0.05f;
        }

        float derivative(const float x) const override
        {
            return std::exp(x) / std::pow((std::exp(x) + 1.0f), 2);
        }
    };
}