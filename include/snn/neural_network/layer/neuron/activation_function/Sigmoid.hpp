#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class Sigmoid final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::sigmoid; }

        [[nodiscard]] auto getName() const -> std::string final { return "sigmoid"; }

    public:
        Sigmoid()
            : ActivationFunction(0, 1)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final { return (tanhf(x / 2.0F) + 1.0F) / 2.0F; }

        [[nodiscard]] auto derivative(const float x) const -> float final
        {
            return (1.0F - powf(tanhf(x / 2.0F), 2.0F)) / 4.0F;
        }
};
}  // namespace snn::internal
