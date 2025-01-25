#pragma once
#include <cmath>
#include <limits>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class GaussianErrorLinearUnit final : public ActivationFunction
{
    private:
        auto getType() const -> activation override { return activation::GELU; }

        auto getName() const -> string override { return "GELU"; }

    public:
        GaussianErrorLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float override
        {
            return x * (tanh(1.702f * x / 2.0f) + 1.0f) / 2.0f;  // approximation
        }

        auto derivative(const float x) const -> float override
        {
            return 1.702f * x * (1.0f - powf(tanhf(1.702f * x / 2.0f), 2.0f)) / 4.0f +
                   (tanh(1.702f * x / 2.0f) + 1.0f) /
                       2.0f;  // 1.702 * x * sigmoid.derivative(x) + sigmoid.function(1.702 * x)
        }
};
}  // namespace snn::internal
