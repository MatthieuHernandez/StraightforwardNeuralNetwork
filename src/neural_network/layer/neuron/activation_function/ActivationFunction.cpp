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

vector<shared_ptr<ActivationFunction>> ActivationFunction::activationFunctions;

ActivationFunction::ActivationFunction(float min, float max)
    : min(min), max(max)
{
}

void ActivationFunction::initialize()
{
    activationFunctions.reserve(6);
    activationFunctions.emplace_back(new Sigmoid());
    activationFunctions.emplace_back(new ImprovedSigmoid());
    activationFunctions.emplace_back(new Tanh());
    activationFunctions.emplace_back(new RectifiedLinearUnit());
    activationFunctions.emplace_back(new Gaussian());
    activationFunctions.emplace_back(new Identity());
}

shared_ptr<ActivationFunction> ActivationFunction::get(activation type)
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
