#include <cmath>
#include <stdexcept>
#include "ActivationFunction.hpp"
#include "Sigmoid.hpp"
#include "ImprovedSigmoid.hpp"
#include "Tanh.hpp"
#include "ReLU.hpp"
#include "Gaussian.hpp"

using namespace std;
using namespace snn;
using namespace internal;

vector<ActivationFunction*> ActivationFunction::activationFunctions;

void ActivationFunction::initialize()
{
	activationFunctions.reserve(4);

	activationFunctions.push_back(new Sigmoid());
	activationFunctions.push_back(new ImprovedSigmoid());
	activationFunctions.push_back(new TanH());
	activationFunctions.push_back(new ReLU());
	activationFunctions.push_back(new Gaussian());
}

ActivationFunction* ActivationFunction::get(activationFunctionType type)
{
	return activationFunctions[type];
}

bool ActivationFunction::operator==(const ActivationFunction& activationFunction) const
{
	return this->getType() == activationFunction.getType();
}

bool ActivationFunction::operator!=(const ActivationFunction& activationFunction) const
{
	return !this->operator==(activationFunction);
}
