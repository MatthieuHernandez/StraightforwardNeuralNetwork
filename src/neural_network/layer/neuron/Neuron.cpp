#include <cmath>
#include "Neuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Neuron::Neuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      activationFunction(model.activationFunction),
      optimizer(optimizer)

{
    this->previousDeltaWeights.resize(model.numberOfWeights, 0);
    this->lastInputs.resize(model.numberOfInputs, 0);
    this->errors.resize(model.numberOfInputs, 0);
    this->outputFunction = ActivationFunction::get(this->activationFunction);
    this->weights.resize(model.numberOfWeights);
    for (auto& w : this->weights)
    {
        w = randomInitializeWeight(model.numberOfWeights);
    }
    this->bias = 1.0f;
}

float Neuron::randomInitializeWeight(int numberOfWeights)
{
    const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfWeights));
    return tools::randomBetween(-valueMax, valueMax);
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
    for (auto& weight : this->weights)
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

NeuralNetworkOptimizer* Neuron::getOptimizer() const
{
    return this->optimizer.get();
}

void Neuron::setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer)
{
    this->optimizer = newOptimizer;
}

bool Neuron::operator==(const Neuron& neuron) const
{
    return typeid(*this).hash_code() == typeid(neuron).hash_code()
        && this->numberOfInputs == neuron.numberOfInputs
        && this->weights == neuron.weights
        && this->bias == neuron.bias
        && this->previousDeltaWeights == neuron.previousDeltaWeights
        && this->lastInputs == neuron.lastInputs
        && this->errors == neuron.errors
        && this->sum == neuron.sum
        && this->activationFunction == neuron.activationFunction
        && this->outputFunction == neuron.outputFunction // not really good
        && *this->optimizer == *neuron.optimizer;
}

bool Neuron::operator!=(const Neuron& Neuron) const
{
    return !(*this == Neuron);
}
