#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

namespace snn::internal
{
    class RectifiedLinearUnit final : public ActivationFunction // WARNING : bad function, if sum < 0 at start, neuron will never learn
    {
    private:
        activation getType() const override { return activation::ReLU; }

    public:
        RectifiedLinearUnit()
            : ActivationFunction(0, +INFINITY)
        {
        }

        float function(const float x) const override
        {
            return (x > 0.0f) ? x : 0.0f;
        }

        float derivative(const float x) const override
        {
            return (x > 0.0f) ? 1.0f : 0.0f;
        }
    };
}