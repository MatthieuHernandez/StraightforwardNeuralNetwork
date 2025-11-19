#pragma once
#include <algorithm>
#include <cmath>

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
            : ActivationFunction(-largeFloat, largeFloat)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final
        {
            const float y = (1.0F / (1.0F + expf(-x))) + (x * 0.05F);  // NOLINT(*magic-numbers)
            return std::clamp(y, this->min, this->max);
        }

        [[nodiscard]] auto derivative(const float x) const -> float final
        {
            return expf(x) / powf((expf(x) + 1.0F), 2);
        }
};
}  // namespace snn::internal
