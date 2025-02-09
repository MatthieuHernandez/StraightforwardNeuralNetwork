#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class Gaussian final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::gaussian; }

        [[nodiscard]] auto getName() const -> std::string final { return "gaussian"; }

    public:
        Gaussian()
            : ActivationFunction(0, 1)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final { return expf(-powf(x, 2)); }

        [[nodiscard]] auto derivative(const float x) const -> float final { return -2 * x * expf(-powf(x, 2)); }
};
}  // namespace snn::internal