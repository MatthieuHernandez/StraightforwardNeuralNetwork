#include <cmath>
#include <boost/serialization/export.hpp>
#include "GateRecurrentUnit.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(GateRecurrentUnit)

GateRecurrentUnit::GateRecurrentUnit(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer)
{
    this->sigmoid = ActivationFunction::get(activation::sigmoid);

    this->resetGateBegin = (this->numberOfInputs + 1) * 1;
    this->resetGateEnd = this->resetGateBegin + this->numberOfInputs;
    this->updateGateBegin = (this->numberOfInputs + 1) * 2;
    this->updateGateEnd = this->updateGateBegin + this->numberOfInputs;
}

float GateRecurrentUnit::output(const vector<float>& inputs, bool temporalReset)
{
    float resetGate = this->resetGateOutput(inputs);
    float updateGate = this->updateGateOutput(inputs);

    this->sum = 0;
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        this->sum += inputs[w] * this->weights[w];
    }
    this->sum += this->previousOutput * resetGate * this->weights[w];
    this->sum += this->bias;
    float output = outputFunction->function(sum);
    output *= updateGate;
    output += this->previousOutput * (1 - updateGate);
    return output;
}

std::vector<float>& GateRecurrentUnit::backOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    return this->errors;
}

void GateRecurrentUnit::train(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);
    throw exception();
}

inline
float GateRecurrentUnit::resetGateOutput(const std::vector<float>& inputs)
{
    float resetGateSum = 0;
    int w;
    for (w = this->resetGateBegin; w < this->resetGateEnd; ++w)
    {
        resetGateSum += inputs[w] * this->weights[w];
    }
    resetGateSum += this->previousOutput * this->weights[w];
    resetGateSum += this->bias;
    return this->sigmoid->function(resetGateSum);
}

inline
float GateRecurrentUnit::updateGateOutput(const std::vector<float>& inputs)
{
    float updateGateSum = 0;
    int w;
    for (w = this->updateGateBegin; w < this->updateGateEnd; ++w)
    {
        updateGateSum += inputs[w] * this->weights[w];
    }
    updateGateSum += this->previousOutput * this->weights[w];
    updateGateSum += this->bias;
    return this->sigmoid->function(updateGateSum);
}

inline
void GateRecurrentUnit::updateResetGateWeights(const float error)
{
    int w;
    for (w = this->resetGateBegin; w < this->resetGateEnd; ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * this->lastInputs[w];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        this->weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
    this->recurrentError = error + this->recurrentError * outputFunction->derivative(this->previousSum) * this->weights[w];

    auto deltaWeights = this->optimizer->learningRate * this->recurrentError * this->previousOutput;
    deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
    this->weights[w] += deltaWeights;
    this->previousDeltaWeights[w] = deltaWeights;
}

inline
void GateRecurrentUnit::updateUpdateGateWeights(const float error)
{
    int w;
    for (w = this->updateGateBegin; w < this->updateGateEnd; ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * this->lastInputs[w];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        this->weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
    this->recurrentError = error + this->recurrentError * outputFunction->derivative(this->previousSum) * this->weights[w];

    auto deltaWeights = this->optimizer->learningRate * this->recurrentError * this->previousOutput;
    deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
    this->weights[w] += deltaWeights;
    this->previousDeltaWeights[w] = deltaWeights;
}

inline
std::vector<float>& GateRecurrentUnit::updateGateBackOutput(float error)
{
     error = error * outputFunction->derivative(this->sum);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    this->updateWeights(error);
    return this->errors;
}

inline
std::vector<float>& GateRecurrentUnit::resetGateBackOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    this->updateWeights(error);
    return this->errors;
}

inline
void GateRecurrentUnit::updateWeights(const std::vector<float>& inputs, float error)
{
    throw exception();
}

inline
void GateRecurrentUnit::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}

int GateRecurrentUnit::isValid() const
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 1)
        return 304;
    return this->Neuron::isValid();
}

bool GateRecurrentUnit::operator==(const Neuron& neuron) const
{
    try
    {
        const auto& n = dynamic_cast<const GateRecurrentUnit&>(neuron);
        return this->Neuron::operator==(neuron)
            && this->lastOutput == n.lastOutput
            && this->previousOutput == n.previousOutput
            && this->recurrentError == n.recurrentError
            && this->previousSum == n.previousSum;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool GateRecurrentUnit::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
