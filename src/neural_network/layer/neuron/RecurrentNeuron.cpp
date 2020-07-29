#include <cmath>
#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(Neuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer),
      numberOfRecurrences(model.numberOfRecurrences),
      numberOfInputs(model.numberOfInputs - this->numberOfRecurrences),
      sizeOfInputs(sizeof(float) * model.numberOfInputs - this->numberOfRecurrences),
      sizeToCopy(sizeof(float) * (model.numberOfRecurrences - 1))
{
    this->previousOutputs.resize(model.numberOfRecurrences, 0);
}

float RecurrentNeuron::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset)
        this->reset();
    lastInputs = inputs;
    float sum = 0;
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        sum += inputs[w] * this->weights[w];
    }
    for (int r = 0; r < this->previousOutputs.size(); ++r, ++w)
    {
        sum += this->previousOutputs[r] * this->weights[w];
    }
    sum += bias;
    lastOutput = sum;
    sum = outputFunction->function(sum);
    this->addPreviousOutput(sum);
    return sum;
}

std::vector<float>& RecurrentNeuron::backOutput(float error)
{
    error = error * outputFunction->derivative(lastOutput);

    this->updateWeights(lastInputs, error);
    this->addPreviousError(error);

    for (int w = 0; w < this->weights.size(); ++w)
    {
        errors[w] = error * this->weights[w];
    }
    return errors;
}

void RecurrentNeuron::train(float error)
{
    error = error * outputFunction->derivative(lastOutput);
    this->updateWeights(lastInputs, error);
    //this->addPreviousError(error);
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
    for (int r = 0; r < this->previousOutputs.size(); ++r, ++w)
    {
        this->recurrentError += error;
        auto deltaWeights = this->optimizer->learningRate * this->recurrentError * this->previousOutputs[r];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        this->weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
}

inline
void RecurrentNeuron::reset()
{
    fill(this->previousOutputs.begin(), this->previousOutputs.end(), 0);
    this->recurrentError = 0;
}

inline
void RecurrentNeuron::addPreviousOutput(float output)
{
    if (this->numberOfRecurrences > 1)
        memcpy(&this->previousOutputs[1], &this->previousOutputs[0], this->sizeToCopy);
    this->previousOutputs[0] = output;
}

inline
void RecurrentNeuron::addPreviousError(float error)
{
    /*if (this->numberOfRecurrences > 1)
        memcpy(&this->previousErrors[1], &this->previousErrors[0], this->sizeToCopy);
    this->previousErrors[0] = error;*/
}


int RecurrentNeuron::isValid() const
{
    if (this->numberOfInputs != static_cast<int>(this->weights.size()) - this->numberOfRecurrences
        || this->previousOutputs.size() != this->numberOfRecurrences)
        return 304;
    return this->Neuron::isValid();
}

int RecurrentNeuron::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

bool RecurrentNeuron::operator==(const Neuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool RecurrentNeuron::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
