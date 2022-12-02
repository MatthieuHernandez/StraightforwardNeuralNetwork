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
    this->lastInputs.push(inputs);
    if (static_cast<int>(this->lastInputs.size()) > this->batchSize)
        this->lastInputs.pop();
    float tmp = 0.0f; // to activate the SIMD optimization
    #pragma omp simd
    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        tmp += inputs[w] * this->weights[w];
    }
    this->sum = tmp + bias;
    return this->outputFunction->function(this->sum);
}

vector<float>& SimpleNeuron::backOutput(float error)
{
    error = error * this->outputFunction->derivative(this->sum);

    #pragma omp simd // seems to do nothing
    for (size_t w = 0; w < this->weights.size(); ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    while (!this->lastInputs.empty())
    {
        this->optimizer->updateWeights(*this, error);
        this->lastInputs.pop();
    }
    return this->errors;
}

void SimpleNeuron::train(float error)
{
    error = error * outputFunction->derivative(this->sum);
    while (!this->lastInputs.empty())
    {
        this->optimizer->updateWeights(*this, error);
        this->lastInputs.pop();
    }
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
