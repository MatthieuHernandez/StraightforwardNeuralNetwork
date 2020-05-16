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

void ActivationFunction::initialize()
{
    activationFunctions.reserve(4);
    activationFunctions.push_back(new Sigmoid());
    activationFunctions.push_back(new ImprovedSigmoid());
    activationFunctions.push_back(new Tanh());
    activationFunctions.push_back(new RectifiedLinearUnit());
    activationFunctions.push_back(new Gaussian());
    activationFunctions.push_back(new Identity());
}

ActivationFunction* ActivationFunction::get(activationFunction type)
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
