#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Gaussian final : public ActivationFunction
    {
    private:
        activationFunction getType() const override { return gaussian; }

    public:
        float function(const float x) const override
        {
            return std::exp(-std::pow(x, 2));
        }

        float derivative(const float x) const override
        {
            return -2 * x * std::exp(-std::pow(x, 2));
        }
    };
}