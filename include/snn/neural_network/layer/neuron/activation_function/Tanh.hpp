#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class Tanh final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::tanh; }

        [[nodiscard]] auto getName() const -> std::string final { return "tanh"; }

    public:
        Tanh()
            : ActivationFunction(-1, 1)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final { return tanhf(x); }

        [[nodiscard]] auto derivative(const float x) const -> float final { return 1 - powf(tanhf(x), 2); }
};
}  // namespace snn::internal
