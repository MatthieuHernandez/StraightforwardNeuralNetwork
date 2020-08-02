#include "../../../../tools/ExtendedExpection.hpp"
#include "ActivationFunction.hpp"
#include "Sigmoid.hpp"
#include "ImprovedSigmoid.hpp"
#include "Tanh.hpp"
#include "ReLU.hpp"
#include "Gaussian.hpp"
#include "Identity.hpp"

using namespace std;
using namespace snn;
using namespace internal;

vector<ActivationFunction*> ActivationFunction::activationFunctions;

ActivationFunction::ActivationFunction(float min, float max)
    : min(min), max(max)
{
}

void ActivationFunction::initialize()
{
    activationFunctions.reserve(6);
    activationFunctions.push_back(new Sigmoid());
    activationFunctions.push_back(new ImprovedSigmoid());
    activationFunctions.push_back(new Tanh());
    activationFunctions.push_back(new RectifiedLinearUnit());
    activationFunctions.push_back(new Gaussian());
    activationFunctions.push_back(new Identity());
}

ActivationFunction* ActivationFunction::get(activation type)
{
    switch (type)
    {
    case activation::sigmoid:
        return activationFunctions[0];
    case activation::iSigmoid:
        return activationFunctions[1];
    case activation::tanh:
        return activationFunctions[2];
    case activation::ReLU:
        return activationFunctions[3];
    case activation::gaussian:
        return activationFunctions[4];
    case activation::identity:
        return activationFunctions[5];
    default:
        throw NotImplementedException("activation");
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
