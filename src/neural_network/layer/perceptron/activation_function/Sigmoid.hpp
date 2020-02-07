#pragma once
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Sigmoid : public ActivationFunction
    {
    private:
        activationFunction getType() const override { return sigmoid; }

    public:
        float function(const float x) const override
        {
            return 1.0f / (1.0f + exp(-x));
        }

        float derivative(const float x) const override
        {
            return exp(-x) / pow((exp(-x) + 1.0f), 2);
        }
    };
}