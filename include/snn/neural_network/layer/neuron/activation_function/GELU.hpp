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
        auto getType() const -> activation final { return activation::GELU; }

        auto getName() const -> string final { return "GELU"; }

    public:
        GaussianErrorLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float final
        {
            return x * (tanh(1.702F * x / 2.0F) + 1.0F) / 2.0F;  // approximation
        }

        auto derivative(const float x) const -> float final
        {
            return 1.702F * x * (1.0F - powf(tanhf(1.702F * x / 2.0F), 2.0F)) / 4.0F +
                   (tanh(1.702F * x / 2.0F) + 1.0F) /
                       2.0F;  // 1.702 * x * sigmoid.derivative(x) + sigmoid.function(1.702 * x)
        }
};
}  // namespace snn::internal
