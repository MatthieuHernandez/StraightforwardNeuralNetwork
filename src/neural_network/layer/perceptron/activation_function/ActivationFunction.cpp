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
			throw std::runtime_error("This type of activation function doesn't exist.");
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
