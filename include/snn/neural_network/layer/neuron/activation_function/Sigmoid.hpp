#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class Sigmoid final : public ActivationFunction
{
    private:
        auto getType() const -> activation override { return activation::sigmoid; }

        auto getName() const -> string override { return "sigmoid"; }

    public:
        Sigmoid()
            : ActivationFunction(0, 1)
        {
        }

        auto function(const float x) const -> float override { return (tanhf(x / 2.0F) + 1.0F) / 2.0F; }

        auto derivative(const float x) const -> float override { return (1.0F - powf(tanhf(x / 2.0F), 2.0F)) / 4.0F; }
};
}  // namespace snn::internal
