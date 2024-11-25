#include <boost/serialization/export.hpp>
#include "SimpleNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

SimpleNeuron::SimpleNeuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, optimizer)
{
}

float SimpleNeuron::output(const vector<float>& inputs)
{
    this->lastInputs.pushBack(inputs);
    float tmp = 0.0f; // to activate the SIMD optimization
    assert(this->weights.size() == inputs.size() + 1);
    size_t w = 0;
    #pragma omp simd
    for (w = 0; w < inputs.size(); ++w)
    {
        tmp += inputs[w] * this->weights[w];
    }
    this->sum = tmp + this->weights[w] * bias;
    return this->outputFunction->function(this->sum);
}

vector<float>& SimpleNeuron::backOutput(float error)
{
    error = error * this->outputFunction->derivative(this->sum);
    assert(this->weights.size() == this->errors.size() + 1);
    #pragma omp simd // seems to do nothing
    for (size_t w = 0; w < this->errors.size(); ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    this->optimizer->updateWeights(*this, error);
    return this->errors;
}

void SimpleNeuron::train(float error)
{
    error = error * this->outputFunction->derivative(this->sum);
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
