#include "GatedRecurrentUnit.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>

namespace snn::internal
{
GatedRecurrentUnit::GatedRecurrentUnit(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      resetGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::sigmoid},
                optimizer),
      updateGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::sigmoid},
                 optimizer),
      outputGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::tanh},
                 optimizer)
{
}

auto GatedRecurrentUnit::output(const std::vector<float>& inputs, bool temporalReset) -> float
{
    if (temporalReset)
    {
        this->reset();
    }
    const float resetGateOutput = this->resetGate.output(inputs, temporalReset);
    this->updateGateOutput = this->updateGate.output(inputs, temporalReset);
    this->outputGate.lastOutput *= resetGateOutput;
    this->outputGateOutput = this->outputGate.output(inputs, temporalReset);

    const float output =
        (1 - this->updateGateOutput) * this->previousOutput + this->updateGateOutput * outputGateOutput;

    this->resetGate.lastOutput = output;
    this->updateGate.lastOutput = output;
    this->outputGate.lastOutput = output;

    return output;
}

auto GatedRecurrentUnit::backOutput(float error) -> std::vector<float>&
{
    const float d3 = error;
    const float d8 = d3 * this->updateGateOutput;
    const float d7 = d3 * this->outputGateOutput;
    const float d9 = d7 + d8;

    this->errors = this->outputGate.backOutput(d8);
    auto e2 = this->updateGate.backOutput(d9);
    const float d13 = this->errors.back();
    const float d16 = d13 * this->previousOutput;
    auto e3 = this->resetGate.backOutput(d16);

    std::ranges::transform(this->errors, e2, this->errors.begin(), std::plus<float>());
    std::ranges::transform(this->errors, e3, this->errors.begin(), std::plus<float>());
    return this->errors;
}

void GatedRecurrentUnit::back(float error)
{
    const float d3 = error;
    const float d8 = d3 * this->updateGateOutput;
    const float d7 = d3 * this->outputGateOutput;
    const float d9 = d7 + d8;

    this->errors = this->outputGate.backOutput(d8);
    auto e2 = this->updateGate.backOutput(d9);
    const float d13 = this->errors.back();
    const float d16 = d13 * this->previousOutput;
    auto e3 = this->resetGate.backOutput(d16);
}

void GatedRecurrentUnit::train()
{
    this->outputGate.train();
    this->updateGate.train();
    this->resetGate.train();
}

auto GatedRecurrentUnit::getWeights() const -> std::vector<float>
{
    std::vector<float> allWeights;
    allWeights.insert(allWeights.end(), this->resetGate.weights.begin(), this->resetGate.weights.end());
    allWeights.insert(allWeights.end(), this->updateGate.weights.begin(), this->updateGate.weights.end());
    allWeights.insert(allWeights.end(), this->outputGate.weights.begin(), this->outputGate.weights.end());
    return allWeights;
}

auto GatedRecurrentUnit::getNumberOfParameters() const -> int
{
    return this->resetGate.getNumberOfParameters() + this->updateGate.getNumberOfParameters() +
           this->outputGate.getNumberOfParameters();
}

auto GatedRecurrentUnit::getNumberOfInputs() const -> int { return this->numberOfInputs; }

inline void GatedRecurrentUnit::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->updateGateOutput = 0;
}

auto GatedRecurrentUnit::isValid() const -> errorType
{
    auto err = resetGate.isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    err = updateGate.isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    err = outputGate.isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    return errorType::noError;
}

auto GatedRecurrentUnit::getOptimizer() const -> NeuralNetworkOptimizer* { return this->resetGate.getOptimizer(); }

void GatedRecurrentUnit::setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer)
{
    this->resetGate.setOptimizer(newOptimizer);
    this->updateGate.setOptimizer(newOptimizer);
    this->outputGate.setOptimizer(newOptimizer);
}

auto GatedRecurrentUnit::operator==(const GatedRecurrentUnit& neuron) const -> bool
{
    return this->numberOfInputs == neuron.numberOfInputs && this->previousOutput == neuron.previousOutput &&
           this->recurrentError == neuron.recurrentError && this->updateGateOutput == neuron.updateGateOutput &&
           this->outputGateOutput == neuron.outputGateOutput && this->resetGate == neuron.resetGate &&
           this->updateGate == neuron.updateGate && this->outputGate == neuron.outputGate;
}

auto GatedRecurrentUnit::operator!=(const GatedRecurrentUnit& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal