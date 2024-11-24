#pragma once
#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
    class Identity final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::identity; }

        string getName() const override { return "identity"; }

    public:
         Identity()
            : ActivationFunction(-INFINITY, +INFINITY)
        {
        }

        float function(const float x) const override
        {
            return x;
        }

        float derivative([[maybe_unused]] const float x) const override
        {
            return 1.0f;
        }
    };
}