#pragma once
#include <cmath>
#include <limits>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class ImprovedSigmoid final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::iSigmoid; }

        [[nodiscard]] auto getName() const -> std::string final { return "iSigmoid"; }

    public:
        ImprovedSigmoid()
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final
        {
            return (1.0F / (1.0F + expf(-x))) + (x * 0.05F);  // NOLINT(*magic-numbers)
        }

        [[nodiscard]] auto derivative(const float x) const -> float final
        {
            return expf(x) / powf((expf(x) + 1.0F), 2);
        }
};
}  // namespace snn::internal