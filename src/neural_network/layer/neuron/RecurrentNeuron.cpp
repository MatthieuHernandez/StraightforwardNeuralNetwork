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
      sizeToCopy(sizeof(float) * (model.numberOfRecurrences-1))
{
    this->recurrences.resize(model.numberOfRecurrences, -1);
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
        sum += inputs[w] * weights[w];
    }
    for (int r = 0; r < this->recurrences.size(); ++r, ++w)
    {
        sum += this->recurrences[r] * weights[w];
    }
    sum += bias;
    lastOutput = sum;
    sum = outputFunction->function(sum);
    this->addNewInputs(sum);
    return sum;
}

inline
void RecurrentNeuron::updateWeights(const std::vector<float>& inputs, float error)
{
    int w;
    for (w = 0; w < inputs.size(); ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * inputs[w];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
    for (int r = 0; r < this->recurrences.size(); ++r, ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * recurrences[r];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
}


inline
void RecurrentNeuron::reset()
{
    fill(this->recurrences.begin(), this->recurrences.end(), -1);
}

inline
void RecurrentNeuron::addNewInputs(float output)
{
    if(this->numberOfRecurrences > 1)
        memcpy(&this->recurrences[1], &this->recurrences[0], this->sizeToCopy);
    this->recurrences[0] = output;
}

int RecurrentNeuron::isValid() const
{
    if (this->numberOfInputs != static_cast<int>(this->weights.size()) - this->numberOfRecurrences
        || this->recurrences.size() != this->numberOfRecurrences)
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
