#pragma once
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class Tanh : public ActivationFunction
    {
    private:

        activationFunction getType() const override { return tanh; }

    public:

        float function(const float x) const override
        {
            return std::tanh(x);
        }

        float derivative(const float x) const override
        {
            return 1 - pow(std::tanh(x), 2);
        }
    };
}