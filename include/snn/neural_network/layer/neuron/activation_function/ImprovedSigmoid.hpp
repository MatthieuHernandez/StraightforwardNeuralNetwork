#pragma once
#include <cmath>
#include <limits>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class ImprovedSigmoid final : public ActivationFunction
{
    private:
        activation getType() const override { return activation::iSigmoid; }

        string getName() const override { return "iSigmoid"; }

    public:
        ImprovedSigmoid()
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        float function(const float x) const override { return 1.0f / (1.0f + std::exp(-x)) + x * 0.05f; }

        float derivative(const float x) const override { return std::exp(x) / powf((std::exp(x) + 1.0f), 2); }
};
}  // namespace snn::internal