#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class Tanh final : public ActivationFunction
{
    private:
        auto getType() const -> activation override { return activation::tanh; }

        auto getName() const -> string override { return "tanh"; }

    public:
        Tanh()
            : ActivationFunction(-1, 1)
        {
        }

        auto function(const float x) const -> float override { return tanhf(x); }

        auto derivative(const float x) const -> float override { return 1 - powf(tanhf(x), 2); }
};
}  // namespace snn::internal
