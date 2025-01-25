#pragma once
#include <limits>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class Identity final : public ActivationFunction
{
    private:
        auto getType() const -> activation override { return activation::identity; }

        auto getName() const -> string override { return "identity"; }

    public:
        Identity()
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float override { return x; }

        auto derivative([[maybe_unused]] const float x) const -> float override { return 1.0F; }
};
}  // namespace snn::internal