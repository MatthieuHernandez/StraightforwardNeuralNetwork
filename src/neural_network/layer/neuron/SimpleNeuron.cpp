#include "SimpleNeuron.hpp"

#include <boost/serialization/export.hpp>
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
    return Neuron::computeOutput();
}

void SimpleNeuron::train() { this->optimizer->updateWeights(*this); }

auto SimpleNeuron::isValid() const -> errorType { return this->Neuron::isValid(); }

auto SimpleNeuron::operator==(const SimpleNeuron& neuron) const -> bool { return this->Neuron::operator==(neuron); }

auto SimpleNeuron::operator!=(const SimpleNeuron& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal
