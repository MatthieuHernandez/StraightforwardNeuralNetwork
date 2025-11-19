#pragma once
#include <algorithm>
#include <limits>

#include "ActivationFunction.hpp"

namespace snn::internal
{
class Identity final : public ActivationFunction
{
    private:
        [[nodiscard]] auto getType() const -> activation final { return activation::identity; }

        [[nodiscard]] auto getName() const -> std::string final { return "identity"; }

    public:
        Identity()
            : ActivationFunction(-largeFloat, largeFloat)
        {
        }

        [[nodiscard]] auto function(const float x) const -> float final { return std::clamp(x, this->min, this->max); }

        [[nodiscard]] auto derivative([[maybe_unused]] const float x) const -> float final { return 1.0F; }
};
}  // namespace snn::internal
