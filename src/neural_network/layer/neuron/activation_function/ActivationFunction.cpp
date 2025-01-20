#include "ActivationFunction.hpp"

#include "ExtendedExpection.hpp"
#include "GELU.hpp"
#include "Gaussian.hpp"
#include "Identity.hpp"
#include "ImprovedSigmoid.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"

using namespace std;
using namespace snn;
using namespace internal;

vector<shared_ptr<ActivationFunction>> ActivationFunction::activationFunctions = initialize();

inline vector<shared_ptr<ActivationFunction>> ActivationFunction::initialize()
{
    vector<shared_ptr<ActivationFunction>> activations;
    activations.reserve(6);
    activations.emplace_back(new Sigmoid());
    activations.emplace_back(new ImprovedSigmoid());
    activations.emplace_back(new Tanh());
    activations.emplace_back(new RectifiedLinearUnit());
    activations.emplace_back(new GaussianErrorLinearUnit());
    activations.emplace_back(new Gaussian());
    activations.emplace_back(new Identity());
    return activations;
}

ActivationFunction::ActivationFunction(float min, float max)
    : min(min),
      max(max)
{
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
        case activation::GELU:
            return activationFunctions[4];
        case activation::gaussian:
            return activationFunctions[5];
        case activation::identity:
            return activationFunctions[6];
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
