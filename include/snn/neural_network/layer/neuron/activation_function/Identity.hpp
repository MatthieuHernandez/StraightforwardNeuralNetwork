#pragma once
#include <limits>
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
            : ActivationFunction(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity())
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