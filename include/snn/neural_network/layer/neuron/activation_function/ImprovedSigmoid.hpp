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
        auto getType() const -> activation final { return activation::iSigmoid; }

        auto getName() const -> string final { return "iSigmoid"; }

    public:
        ImprovedSigmoid()
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float final { return 1.0F / (1.0F + std::exp(-x)) + x * 0.05F; }

        auto derivative(const float x) const -> float final { return std::exp(x) / powf((std::exp(x) + 1.0F), 2); }
};
}  // namespace snn::internal