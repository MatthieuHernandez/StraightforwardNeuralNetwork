#include "RecurrentNeuron.hpp"

#include <boost/serialization/export.hpp>

namespace snn::internal
{
RecurrentNeuron::RecurrentNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, optimizer)
{
}

auto RecurrentNeuron::output(const std::vector<float>& inputs, bool temporalReset) -> float
{
    if (temporalReset)
    {
        this->reset();
    }
    this->lastInputs.pushBack(inputs, this->lastOutput, this->bias);
    this->lastOutput = Neuron::computeOutput();
    return this->lastOutput;
}

void RecurrentNeuron::train() { this->optimizer->updateWeights(*this); }

inline void RecurrentNeuron::reset()
{
    this->recurrentError = 0;
    this->previousSum = 0;
}

auto RecurrentNeuron::isValid() const -> errorType
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 2)
    {
        return errorType::recurrentNeuronWrongNumberOfWeight;
    }
    return this->Neuron::isValid();
}

auto RecurrentNeuron::operator==(const RecurrentNeuron& neuron) const -> bool
{
    return this->Neuron::operator==(neuron) && this->lastOutput == neuron.lastOutput &&
           this->recurrentError == neuron.recurrentError && this->previousSum == neuron.previousSum;
}

auto RecurrentNeuron::operator!=(const RecurrentNeuron& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal
