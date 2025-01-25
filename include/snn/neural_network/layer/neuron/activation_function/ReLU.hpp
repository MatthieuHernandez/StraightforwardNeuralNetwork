#pragma once
#include <limits>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class RectifiedLinearUnit final : public ActivationFunction
{
    private:
        auto getType() const -> activation override { return activation::ReLU; }

        auto getName() const -> string override { return "ReLU"; }

    public:
        RectifiedLinearUnit()
            : ActivationFunction(0, std::numeric_limits<float>::infinity())
        {
        }

        auto function(const float x) const -> float override { return (x > 0.0f) ? x : 0.0f; }

        auto derivative(const float x) const -> float override { return (x > 0.0f) ? 1.0f : 0.0f; }
};
}  // namespace snn::internal