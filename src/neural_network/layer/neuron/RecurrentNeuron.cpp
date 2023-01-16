#include <cmath>
#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(RecurrentNeuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, optimizer)
{
}

#ifdef _MSC_VER
#pragma warning(disable:4701)
#endif
float RecurrentNeuron::output(const vector<float>& inputs, bool temporalReset)
{
    if (temporalReset)
        this->reset();
    this->lastInputs.push(inputs);
    if (static_cast<int>(this->lastInputs.size()) > this->batchSize)
        this->lastInputs.pop();
    this->previousSum = this->sum;
    this->previousOutput = this->lastOutput;
    this->sum = 0;
    int w = 0;
    float tmp = 0.0f; // to activate the SIMD optimization
    #pragma omp simd
    for (w = 0; w < (int)inputs.size(); ++w)
    {
        tmp += inputs[w] * this->weights[w];
    }
    this->sum = tmp + this->previousOutput * this->weights[w] + this->bias;
    float output = outputFunction->function(sum);
    this->lastOutput = output;
    return output;
    #ifdef _MSC_VER
    #pragma warning(default:4701)
    #endif
}

vector<float>& RecurrentNeuron::backOutput(float error)
{
    error = error * this->outputFunction->derivative(this->sum);
    const auto numberOfWeights = this->weights.size();
    const size_t batchSize_ = this->batchSize;
    #pragma omp simd // seems to do nothing
    for (size_t w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    while (!this->lastInputs.empty())
    {
        if (this->previousDeltaWeights.empty())
            this->previousDeltaWeights.push(vector<float>(numberOfWeights, 0.0f));
        this->optimizer->updateWeights(*this, error);
        if (this->previousDeltaWeights.size() > batchSize_)
        this->previousDeltaWeights.pop();
        this->lastInputs.pop();
    }
    return this->errors;
}

void RecurrentNeuron::train(float error)
{
    error = error * this->outputFunction->derivative(this->sum);
    const auto numberOfWeights = this->weights.size();
    const size_t batchSize_ = this->batchSize;
    while (!this->lastInputs.empty())
    {
        if (this->previousDeltaWeights.empty())
            this->previousDeltaWeights.push(vector<float>(numberOfWeights, 0.0f));
        this->optimizer->updateWeights(*this, error);
        if (this->previousDeltaWeights.size() > batchSize_)
        this->previousDeltaWeights.pop();
        this->lastInputs.pop();
    }
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

bool RecurrentNeuron::operator==(const RecurrentNeuron& neuron) const
{
        return this->Neuron::operator==(neuron)
            && this->lastOutput == neuron.lastOutput
            && this->previousOutput == neuron.previousOutput
            && this->recurrentError == neuron.recurrentError
            && this->previousSum == neuron.previousSum;
}

bool RecurrentNeuron::operator!=(const RecurrentNeuron& neuron) const
{
    return !(*this == neuron);
}
