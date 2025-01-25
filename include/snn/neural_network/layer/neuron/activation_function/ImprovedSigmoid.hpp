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
        auto getType() const -> activation override { return activation::iSigmoid; }

        auto getName() const -> string override { return "iSigmoid"; }

    public:
        ImprovedSigmoid()
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float override { return 1.0f / (1.0f + std::exp(-x)) + x * 0.05f; }

        auto derivative(const float x) const -> float override { return std::exp(x) / powf((std::exp(x) + 1.0f), 2); }
};
}  // namespace snn::internal