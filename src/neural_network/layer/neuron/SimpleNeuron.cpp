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
    this->lastInputs.pushBack(inputs);
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
    const auto numberOfWeights = this->weights.size();
    #pragma omp simd // seems to do nothing
    for (size_t w = 0; w < numberOfWeights; ++w)
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
