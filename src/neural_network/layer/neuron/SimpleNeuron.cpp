#include "SimpleNeuron.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <numeric>
#include <utility>

namespace snn::internal
{
SimpleNeuron::SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, std::move(optimizer))
{
}

auto SimpleNeuron::output(const std::vector<float>& inputs) -> float
{
    this->lastInputs.pushBack(inputs, this->bias);
    const auto fullInput = *this->lastInputs.getBack();
    assert(this->weights.size() == fullInput.size());
    const auto sum = std::inner_product(weights.begin(), weights.end(), fullInput.begin(), 0.0F);
    this->lastSum.pushBack(sum);
    return this->outputFunction->function(sum);
}

auto SimpleNeuron::backOutput(float error) -> std::vector<float>&
{
    const auto& sum = *this->lastSum.popFront();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
    assert(this->weights.size() == this->errors.size() + 1);
    std::ranges::transform(errors, weights, errors.begin(), [e](float, float w) -> float { return e * w; });
    return this->errors;
}

void SimpleNeuron::back(float error)
{
    const auto& sum = *this->lastSum.popFront();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
}

void SimpleNeuron::train() { this->optimizer->updateWeights(*this); }

auto SimpleNeuron::isValid() const -> errorType { return this->Neuron::isValid(); }

auto SimpleNeuron::operator==(const SimpleNeuron& neuron) const -> bool { return this->Neuron::operator==(neuron); }

auto SimpleNeuron::operator!=(const SimpleNeuron& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal
