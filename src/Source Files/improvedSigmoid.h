#pragma once
#include "activationFunction.h"
#include <cstdio>

class ImprovedSigmoid : public ActivationFunction
{
private :

	activationFunctionType getType() const override { return iSigmoid; }

public:

	float function(const float x) const override
	{
		return 1.0f / (1.0f + exp(-x)) + 0.05f * x;
	}

	float derivative(const float x) const override
	{
		return exp(x) / pow((exp(x) + 1.0f), 2);
	}
};
