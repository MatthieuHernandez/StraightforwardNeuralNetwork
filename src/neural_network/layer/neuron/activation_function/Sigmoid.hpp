#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Sigmoid final : public ActivationFunction
    {
    private:
        activationFunction getType() const override { return sigmoid; }

    public:
        float function(const float x) const override
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        float derivative(const float x) const override
        {
            return std::exp(-x) / std::pow((std::exp(-x) + 1.0f), 2);
        }
    };
}