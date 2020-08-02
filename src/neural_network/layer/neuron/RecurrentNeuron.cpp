#include <cmath>
#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(Neuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer)
{
}

float RecurrentNeuron::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset)
        this->reset();
    this->lastInputs = inputs;
    this->previousSum = this->sum;
    this->previousOutput = this->lastOutput;
    this->sum = 0;
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        this->sum += inputs[w] * this->weights[w];
    }
    this->sum += this->previousOutput * this->weights[w] + this->bias;
    float output = outputFunction->function(sum);
    this->lastOutput = output;
    return output;
}

std::vector<float>& RecurrentNeuron::backOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    return this->errors;
}

void RecurrentNeuron::train(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);
}

inline
void RecurrentNeuron::updateWeights(const std::vector<float>& inputs, float error)
{
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * inputs[w];
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
void RecurrentNeuron::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}

int RecurrentNeuron::isValid() const
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 1)
        return 304;
    return this->Neuron::isValid();
}

bool RecurrentNeuron::operator==(const Neuron& neuron) const
{
    try
    {
        const auto& n = dynamic_cast<const RecurrentNeuron&>(neuron);
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

bool RecurrentNeuron::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
