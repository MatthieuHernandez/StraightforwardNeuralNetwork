#pragma once
#include "activationFunction.h"

class TanH : public ActivationFunction
{
private :

	activationFunctionType getType() const override { return tanH; }

public:

	float function(const float x) const override
	{
		return tanh(x);
	}

	float derivative(const float x) const override
	{
		return 1 - pow(tanh(x), 2);
	}
};
