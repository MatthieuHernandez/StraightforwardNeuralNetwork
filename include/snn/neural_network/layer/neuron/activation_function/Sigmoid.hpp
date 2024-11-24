#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
    class Sigmoid final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::sigmoid; }

        string getName() const override { return "sigmoid"; }

    public:
        Sigmoid()
            : ActivationFunction(0, 1)
        {
        }

        float function(const float x) const override
        {
            return (tanhf(x / 2.0f) + 1.0f) / 2.0f;
        }

        float derivative(const float x) const override
        {
            return (1.0f - powf(tanhf(x / 2.0f), 2.0f)) / 4.0f;
        }
    };
}
