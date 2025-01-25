#pragma once
#include <limits>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class RectifiedLinearUnit final : public ActivationFunction
{
    private:
        auto getType() const -> activation final { return activation::ReLU; }

        auto getName() const -> string final { return "ReLU"; }

    public:
        RectifiedLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float final { return (x > 0.0F) ? x : 0.0F; }

        auto derivative(const float x) const -> float final { return (x > 0.0F) ? 1.0F : 0.0F; }
};
}  // namespace snn::internal