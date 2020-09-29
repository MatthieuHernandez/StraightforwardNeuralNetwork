#include <boost/serialization/export.hpp>
#include "SimpleNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(SimpleNeuron)

SimpleNeuron::SimpleNeuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer)
{
}

float SimpleNeuron::output(const vector<float>& inputs)
{
    this->lastInputs = inputs;
    this->sum = 0;
    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        this->sum += inputs[w] * weights[w];
    }
    this->sum += bias;
    return outputFunction->function(this->sum);
}

vector<float>& SimpleNeuron::backOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);

    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        this->errors[w] = error * weights[w];
    }
    this->updateWeights(error);
    return this->errors;
}

void SimpleNeuron::train(float error)
{
    error = error * outputFunction->derivative(this->sum);

    this->updateWeights(error);
}

void SimpleNeuron::updateWeights(const float error)
{
    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * this->lastInputs[w];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
}

int SimpleNeuron::isValid() const
{
    return this->Neuron::isValid();
}

bool SimpleNeuron::operator==(const BaseNeuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool SimpleNeuron::operator!=(const BaseNeuron& neuron) const
{
    return !(*this == neuron);
}
