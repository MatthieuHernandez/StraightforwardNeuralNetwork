#include "GatedRecurrentUnit.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

GatedRecurrentUnit::GatedRecurrentUnit(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs),
      resetGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::sigmoid},
                optimizer),
      updateGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::sigmoid},
                 optimizer),
      outputGate({model.numberOfInputs, model.batchSize, model.numberOfWeights, model.bias, activation::tanh},
                 optimizer)
{
}

float GatedRecurrentUnit::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset) this->reset();
    float resetGateOutput = this->resetGate.output(inputs, temporalReset);
    this->updateGateOutput = this->updateGate.output(inputs, temporalReset);
    this->outputGate.lastOutput *= resetGateOutput;
    this->outputGateOutput = this->outputGate.output(inputs, temporalReset);

    float output = (1 - this->updateGateOutput) * this->previousOutput + this->updateGateOutput * outputGateOutput;

    this->resetGate.lastOutput = output;
    this->updateGate.lastOutput = output;
    this->outputGate.lastOutput = output;

    return output;
}

std::vector<float>& GatedRecurrentUnit::backOutput(float error)
{
    float d3 = error;
    float d8 = d3 * this->updateGateOutput;
    float d7 = d3 * this->outputGateOutput;
    float d9 = d7 + d8;

    this->errors = this->outputGate.backOutput(d8);
    auto e2 = this->updateGate.backOutput(d9);
    float d13 = this->errors.back();
    float d16 = d13 * this->previousOutput;
    auto e3 = this->resetGate.backOutput(d16);

    ranges::transform(this->errors, e2, this->errors.begin(), plus<float>());
    ranges::transform(this->errors, e3, this->errors.begin(), plus<float>());
    return this->errors;
}

void GatedRecurrentUnit::train(float error)
{
    float d3 = error;
    float d8 = d3 * this->updateGateOutput;
    float d7 = d3 * this->outputGateOutput;
    float d9 = d7 + d8;

    this->errors = this->outputGate.backOutput(d8);
    auto e2 = this->updateGate.backOutput(d9);
    float d13 = this->errors.back();
    float d16 = d13 * this->previousOutput;
    auto e3 = this->resetGate.backOutput(d16);
}

vector<float> GatedRecurrentUnit::getWeights() const
{
    vector<float> allWeights;
    allWeights.insert(allWeights.end(), this->resetGate.weights.begin(), this->resetGate.weights.end());
    allWeights.insert(allWeights.end(), this->updateGate.weights.begin(), this->updateGate.weights.end());
    allWeights.insert(allWeights.end(), this->outputGate.weights.begin(), this->outputGate.weights.end());
    return allWeights;
}

int GatedRecurrentUnit::getNumberOfParameters() const
{
    return this->resetGate.getNumberOfParameters() + this->updateGate.getNumberOfParameters() +
           this->outputGate.getNumberOfParameters();
}

int GatedRecurrentUnit::getNumberOfInputs() const { return this->numberOfInputs; }

inline void GatedRecurrentUnit::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->updateGateOutput = 0;
}

int GatedRecurrentUnit::isValid() const
{
    auto err = resetGate.isValid();
    if (err != 0) return err;
    err = updateGate.isValid();
    if (err != 0) return err;
    err = outputGate.isValid();
    if (err != 0) return err;
    return 0;
}

NeuralNetworkOptimizer* GatedRecurrentUnit::getOptimizer() const { return this->resetGate.getOptimizer(); }

void GatedRecurrentUnit::setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer)
{
    this->resetGate.setOptimizer(newOptimizer);
    this->updateGate.setOptimizer(newOptimizer);
    this->outputGate.setOptimizer(newOptimizer);
}

bool GatedRecurrentUnit::operator==(const GatedRecurrentUnit& neuron) const
{
    return this->numberOfInputs == neuron.numberOfInputs && this->previousOutput == neuron.previousOutput &&
           this->recurrentError == neuron.recurrentError && this->updateGateOutput == neuron.updateGateOutput &&
           this->outputGateOutput == neuron.outputGateOutput && this->resetGate == neuron.resetGate &&
           this->updateGate == neuron.updateGate && this->outputGate == neuron.outputGate;
}

bool GatedRecurrentUnit::operator!=(const GatedRecurrentUnit& neuron) const { return !(*this == neuron); }
