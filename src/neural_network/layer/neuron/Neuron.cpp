#include "Neuron.hpp"

#include <stdexcept>

#include "../../../tools/Tools.hpp"

namespace snn::internal
{
Neuron::Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      numberOfUses(model.numberOfUses),
      bias(model.bias),
      activationFunction(model.activationFunction),
      optimizer(std::move(optimizer))

{
    this->outputFunction = ActivationFunction::get(this->activationFunction);
    this->weights.resize(model.numberOfWeights);
    for (auto& weight : this->weights)
    {
        weight = randomInitializeWeight(model.numberOfWeights);
    }
    this->weights.back() = std::abs(this->weights.back());
    this->resetLearningVariables();
}

auto Neuron::randomInitializeWeight(int numberOfWeights) -> float
{
    const float valueMax = 2.4F / sqrtf(static_cast<float>(numberOfWeights));
    return tools::randomBetween(-valueMax, valueMax);
}

auto Neuron::isValid() const -> errorType
{
    const auto outlier_float = 100000.0F;
    const size_t outlier_size = 1000000;
    if (this->bias < -outlier_float || this->bias > outlier_float)
    {
        return errorType::neuronWrongBias;
    }

    if (this->weights.empty() || this->weights.size() > outlier_size)
    {
        return errorType::neuronTooMuchWeigths;
    }
    for (const auto& weight : this->weights)
    {
        if (weight < -outlier_float || weight > outlier_float)
        {
            return errorType::neuronWrongWeight;
        }
    }
    return errorType::noError;
}

auto Neuron::getWeights() const -> std::vector<float> { return this->weights; }

void Neuron::setWeights(std::vector<float> w)
{
    if (this->weights.size() != w.size())
    {
        throw std::runtime_error("The size of weights does not match.");
    }
    this->weights = std::move(w);
}

auto Neuron::getNumberOfParameters() const -> int { return static_cast<int>(this->weights.size()); }

auto Neuron::getNumberOfInputs() const -> int { return this->numberOfInputs; }

auto Neuron::getOptimizer() const -> NeuralNetworkOptimizer* { return this->optimizer.get(); }

void Neuron::setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer)
{
    this->optimizer = std::move(newOptimizer);
}

void Neuron::resetLearningVariables()
{
    this->deltaWeights.assign(this->weights.size(), 0.0F);
    this->errors.assign(this->numberOfInputs, 0.0F);
    this->lastInputs.initialize(this->numberOfUses, this->numberOfInputs);
    this->lastError.initialize(this->numberOfUses);
    this->lastSum.initialize(this->numberOfUses);
}

auto Neuron::operator==(const Neuron& neuron) const -> bool
{
    return typeid(*this).hash_code() == typeid(neuron).hash_code() && this->numberOfInputs == neuron.numberOfInputs &&
           this->weights == neuron.weights && this->bias == neuron.bias && this->deltaWeights == neuron.deltaWeights &&
           this->lastInputs == neuron.lastInputs && this->lastError == neuron.lastError &&
           this->lastSum == neuron.lastSum && this->errors == neuron.errors &&
           this->activationFunction == neuron.activationFunction && this->outputFunction == neuron.outputFunction &&
           *this->optimizer == *neuron.optimizer;
}
}  // namespace snn::internal
