﻿#pragma once
#include <cmath>
#include "ActivationFunction.hpp"

using namespace std;

namespace snn::internal
{
    class Gaussian final : public ActivationFunction
    {
    private:
        activation getType() const override { return activation::gaussian; }

        string getName() const override { return "gaussian"; }

    public:
        Gaussian()
            : ActivationFunction(0, 1)
        {
        }

        float function(const float x) const override
        {
            return std::exp(-powf(x, 2));
        }

        float derivative(const float x) const override
        {
            return -2 * x * std::exp(-powf(x, 2));
        }
    };
}