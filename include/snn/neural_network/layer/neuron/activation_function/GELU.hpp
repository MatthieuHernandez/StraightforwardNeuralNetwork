#pragma once
#include <cmath>
#include <limits>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class GaussianErrorLinearUnit final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::GELU; }

        [[nodiscard]] auto getName() const -> std::string final { return "GELU"; }

    public:
        GaussianErrorLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final
        {
            return x * (tanh(1.702F * x / 2.0F) + 1.0F) / 2.0F;  // NOLINT(*magic-numbers)
        }

        [[nodiscard]] auto derivative(const float x) const -> float final
        {
            // 1.702 * x * sigmoid.derivative(x) + sigmoid.function(1.702 * x)
            return (1.702F * x * (1.0F - powf(tanhf(1.702F * x / 2.0F), 2.0F)) / 4.0F) +  // NOLINT(*magic-numbers)
                   ((tanh(1.702F * x / 2.0F) + 1.0F) / 2.0F);                             // NOLINT(*magic-numbers)
        }
};
}  // namespace snn::internal
