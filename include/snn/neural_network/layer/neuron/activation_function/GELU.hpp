#pragma once
#include <limits>
#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
    class GaussianErrorLinearUnit final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::GELU; }

        string getName() const override { return "GELU"; }

    public:
        GaussianErrorLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        float function(const float x) const override
        {
            return x * (tanh(1.702f * x / 2.0f) + 1.0f) / 2.0f; // approximation
        }

        float derivative(const float x) const override
        {
            return 1.702f * x * (1.0f - powf(tanhf(1.702f * x / 2.0f), 2.0f)) / 4.0f
                + (tanh(1.702f * x / 2.0f) + 1.0f) / 2.0f; // 1.702 * x * sigmoid.derivative(x) + sigmoid.function(1.702 * x)
        }
    };
}
