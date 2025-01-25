#include "Neuron.hpp"

namespace snn::internal
{
Neuron::Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      batchSize(model.batchSize),
      bias(model.bias),
      activationFunction(model.activationFunction),
      optimizer(std::move(optimizer))

{
    this->errors.resize(model.numberOfInputs, 0);
    this->outputFunction = ActivationFunction::get(this->activationFunction);
    this->weights.resize(model.numberOfWeights);
    for (auto& weight : this->weights)
    {
        weight = randomInitializeWeight(model.numberOfWeights);
    }
    this->weights.back() = std::abs(this->weights.back());
    this->lastInputs.initialize(this->batchSize, model.numberOfInputs);
    this->previousDeltaWeights.initialize(this->batchSize, model.numberOfWeights);
}

auto Neuron::randomInitializeWeight(int numberOfWeights) -> float
{
    const float valueMax = 2.4F / sqrtf(static_cast<float>(numberOfWeights));
    return tools::randomBetween(-valueMax, valueMax);
}

auto Neuron::isValid() const -> ErrorType
{
    const auto outlier_float = 100000.0F;
    const size_t outlier_size = 1000000;
    if (this->bias < -outlier_float || this->bias > outlier_float)
    {
        return ErrorType::neuronWrongBias;
    }

    if (this->weights.empty() || this->weights.size() > outlier_size)
    {
        return ErrorType::neuronTooMuchWeigths;
    }
    for (const auto& weight : this->weights)
    {
        if (weight < -outlier_float || weight > outlier_float)
        {
            return ErrorType::neuronWrongWeight;
        }
    }
    return ErrorType::noError;
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

auto Neuron::operator==(const Neuron& neuron) const -> bool
{
    return typeid(*this).hash_code() == typeid(neuron).hash_code() && this->numberOfInputs == neuron.numberOfInputs &&
           this->weights == neuron.weights && this->bias == neuron.bias &&
           this->previousDeltaWeights == neuron.previousDeltaWeights && this->lastInputs == neuron.lastInputs &&
           this->errors == neuron.errors && this->sum == neuron.sum &&
           this->activationFunction == neuron.activationFunction &&
           this->outputFunction == neuron.outputFunction  // not really good
           && *this->optimizer == *neuron.optimizer;
}

auto Neuron::operator!=(const Neuron& Neuron) const -> bool { return !(*this == Neuron); }
}  // namespace snn::internal
