#include <cmath>
#include <typeinfo>
#include "Neuron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Neuron::Neuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : numberOfInputs(model.numberOfInputs),
      activationFunction(model.activationFunction),
      optimizer(optimizer)
{
    this->previousDeltaWeights.resize(model.numberOfWeights, 0);
    this->lastInputs.resize(model.numberOfInputs, 0);
    this->errors.resize(model.numberOfWeights, 0);
    this->outputFunction = ActivationFunction::get(this->activationFunction);
    this->weights.resize(model.numberOfWeights);
    for (auto& w : weights)
    {
        w = randomInitializeWeight(model.numberOfWeights);
    }
    this->bias = 1.0f;
}

float Neuron::randomInitializeWeight(int numberOfInputs) const
{
    const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfInputs));
    return Tools::randomBetween(-valueMax, valueMax);
}

int Neuron::isValid() const
{
    if (this->bias != 1.0f)
        return 301;

    if (this->weights.empty()
        || this->weights.size() > 1000000)
    {
        return 302;
    }
    for (auto& weight : weights)
        if (weight < -100000 || weight > 10000)
            return 303;

    return 0;
}

vector<float> Neuron::getWeights() const
{
    return this->weights;
}

int Neuron::getNumberOfParameters() const
{
    return static_cast<int>(this->weights.size());
}

int Neuron::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

bool Neuron::operator==(const BaseNeuron& neuron) const
{
    try
    {
        const auto& n = dynamic_cast<const Neuron&>(neuron);
        return this->BaseNeuron::operator==(neuron)
            && this->numberOfInputs == n.numberOfInputs
            && this->weights == n.weights
            && this->bias == n.bias
            && this->previousDeltaWeights == n.previousDeltaWeights
            && this->lastInputs == n.lastInputs
            && this->errors == n.errors
            && this->sum == n.sum
            && this->activationFunction == n.activationFunction
            && this->outputFunction == n.outputFunction // not really good
            && *this->optimizer == *n.optimizer;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool Neuron::operator!=(const BaseNeuron& Neuron) const
{
    return !(*this == Neuron);
}
