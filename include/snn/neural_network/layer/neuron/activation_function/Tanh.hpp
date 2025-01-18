#pragma once
#include <cmath>

#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
class Tanh final : public ActivationFunction
{
    private:
        activation getType() const override { return activation::tanh; }

        string getName() const override { return "tanh"; }

    public:
        Tanh()
            : ActivationFunction(-1, 1)
        {
        }

        float function(const float x) const override { return tanhf(x); }

        float derivative(const float x) const override { return 1 - powf(tanhf(x), 2); }
};
}  // namespace snn::internal
