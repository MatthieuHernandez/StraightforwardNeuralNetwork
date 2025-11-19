#pragma once
#include <algorithm>
#include <limits>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class RectifiedLinearUnit final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::ReLU; }

        [[nodiscard]] auto getName() const -> std::string final { return "ReLU"; }

    public:
        RectifiedLinearUnit()
            : ActivationFunction(0, largeFloat)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final { return std::clamp(x, 0.0F, this->max); }

        [[nodiscard]] auto derivative(const float x) const -> float final { return (x > 0.0F) ? 1.0F : 0.0F; }
};
}  // namespace snn::internal
