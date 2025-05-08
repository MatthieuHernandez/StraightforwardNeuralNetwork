#pragma once
#include <limits>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class LeakyRectifiedLinearUnit final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::LeakyReLU; }

        [[nodiscard]] auto getName() const -> std::string final { return "LeakyReLU"; }

    public:
        LeakyRectifiedLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        static constexpr float negativeSlopeAngle = 0.001F;

        [[nodiscard]] auto function(const float x) const -> float final
        {
            return (x > 0.0F) ? x : negativeSlopeAngle * x;
        }

        [[nodiscard]] auto derivative(const float x) const -> float final
        {
            return (x > 0.0F) ? 1.0F : negativeSlopeAngle;
        }
};
}  // namespace snn::internal
