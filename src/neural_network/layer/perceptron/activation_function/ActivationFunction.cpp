#include <cmath>
#include "activationFunction.hpp"
#include "sigmoid.hpp"
#include "improvedSigmoid.hpp"
#include "tanh.hpp"
#include "ReLU.hpp"
#include "gaussian.hpp"
#include <exception>

using namespace std;
using namespace snn;
using namespace internal;

//vector<ActivationFunction*> ActivationFunction::listOfActivationFunction;

void ActivationFunction::initialize()
{
	/*listOfActivationFunction.reserve(4);

	listOfActivationFunction.push_back(new Sigmoid());
	listOfActivationFunction.push_back(new ImprovedSigmoid());
	listOfActivationFunction.push_back(new TanH());
	listOfActivationFunction.push_back(new ReLU());
	listOfActivationFunction.push_back(new Gaussian());*/
}

ActivationFunction* ActivationFunction::create(activationFunctionType type)
{
	switch (type)
	{
		case sigmoid:
			return new Sigmoid();
		case iSigmoid:
			return new ImprovedSigmoid();
		case tanH:
			return new TanH();
		case reLU:
			return new ReLU();
		case gaussian:
			return new Gaussian();
		default:
			throw std::exception();
	}
}

bool ActivationFunction::operator==(const ActivationFunction& activationFunction) const
{
	return this->getType() == activationFunction.getType();
}

bool ActivationFunction::operator!=(const ActivationFunction& activationFunction) const
{
	return !this->operator==(activationFunction);
}
