#include <cmath>
#include "Neuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Neuron::Neuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      batchSize(model.batchSize),
      activationFunction(model.activationFunction),
      optimizer(optimizer)

{
    this->errors.resize(model.numberOfInputs, 0);
    this->outputFunction = ActivationFunction::get(this->activationFunction);
    this->weights.resize(model.numberOfWeights);
    for (auto& w : this->weights)
    {
        w = randomInitializeWeight(model.numberOfWeights);
    }
    this->weights.back() = std::abs(this->weights.back());
    this->bias = model.bias;
    this->lastInputs.initialize(this->batchSize, model.numberOfInputs);
    this->previousDeltaWeights.initialize(this->batchSize, model.numberOfWeights);
}

float Neuron::randomInitializeWeight(int numberOfWeights)
{
    const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfWeights));
    return tools::randomBetween(-valueMax, valueMax);
}

int Neuron::isValid() const
{
    if (this->bias < -100000.0f || this->bias > 10000.0f)
        return 301;

    if (this->weights.empty()
        || this->weights.size() > 1000000)
    {
        return 302;
    }
    for (auto& weight : this->weights)
        if (weight < -100000.0f || weight > 10000.0f)
            return 303;

    return 0;
}

vector<float> Neuron::getWeights() const
{
    return this->weights;
}

void Neuron::setWeights(std::vector<float> w)
{
    if (this->weights.size() != w.size())
        throw std::runtime_error("The size of weights does not match.");
    this->weights = std::move(w);
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
    this->optimizer = std::move(newOptimizer);
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
