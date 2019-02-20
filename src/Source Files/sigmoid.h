#pragma once
#include "activationFunction.h"
#include <limits>
#include <string>
#include <cassert>

class Sigmoid : public ActivationFunction
{
private :

	activationFunctionType getType() const override { return sigmoid; }


public:

	float function(const float x) const override
	{
		float result = 1.0f / (1.0f + exp(-x));
		if (isnan(result))
		{
			if (x > 0)
				return 1;
			return 0;
		}
		return result;
	}

	float derivative(const float x) const override
	{
		float result = exp(-x) / pow((exp(-x) + 1.0f), 2);
		if (isnan(result))
			return 0.0f;
		assert(!isnan(result));
		return result;
	}
};
