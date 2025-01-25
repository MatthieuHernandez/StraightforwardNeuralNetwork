#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class Gaussian final : public ActivationFunction
{
    private:
        auto getType() const -> activation final { return activation::gaussian; }

        auto getName() const -> string final { return "gaussian"; }

    public:
        Gaussian()
            : ActivationFunction(0, 1)
        {
        }

        auto function(const float x) const -> float final { return std::exp(-powf(x, 2)); }

        auto derivative(const float x) const -> float final { return -2 * x * std::exp(-powf(x, 2)); }
};
}  // namespace snn::internal