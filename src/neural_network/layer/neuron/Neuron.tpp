#include "Neuron.hpp"

template <class Derived>
Neuron<Derived>::Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : BaseNeuron<Derived>(optimizer),
      numberOfInputs(model.numberOfInputs),
      activationFunction(model.activationFunction)
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

template <class Derived>
float Neuron<Derived>::randomInitializeWeight(int numberOfWeights)
{
    const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfWeights));
    return tools::randomBetween(-valueMax, valueMax);
}

template <class Derived>
int Neuron<Derived>::isValid() const
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

template <class Derived>
std::vector<float> Neuron<Derived>::getWeights() const
{
    return this->weights;
}

template <class Derived>
int Neuron<Derived>::getNumberOfParameters() const
{
    return static_cast<int>(this->weights.size());
}

template <class Derived>
int Neuron<Derived>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <class Derived>
bool Neuron<Derived>::operator==(const Neuron& neuron) const
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

template <class Derived>
bool Neuron<Derived>::operator!=(const Neuron& Neuron) const
{
    return !(*this == Neuron);
}
