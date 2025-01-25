#pragma once
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
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float final { return x; }

        auto derivative([[maybe_unused]] const float x) const -> float final { return 1.0F; }
};
}  // namespace snn::internal