#include <boost/serialization/export.hpp>
#include "SimpleNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(SimpleNeuron)

SimpleNeuron::SimpleNeuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
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
    return this->outputFunction->function(this->sum);
}

vector<float>& SimpleNeuron::backOutput(float error)
{
    error = error * this->outputFunction->derivative(this->sum);

    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        this->errors[w] = error * weights[w];
    }
    this->optimizer->updateWeights(*this, error);
    return this->errors;
}

void SimpleNeuron::train(float error)
{
    error = error * outputFunction->derivative(this->sum);

    this->optimizer->updateWeights(*this, error);
}

int SimpleNeuron::isValid() const
{
    return this->Neuron::isValid();
}

bool SimpleNeuron::operator==(const SimpleNeuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool SimpleNeuron::operator!=(const SimpleNeuron& neuron) const
{
    return !(*this == neuron);
}
