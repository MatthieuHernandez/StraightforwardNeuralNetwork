#include "RecurrentNeuron.hpp"

#include <boost/serialization/export.hpp>
#include <cmath>

namespace snn::internal
{
RecurrentNeuron::RecurrentNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, optimizer)
{
}

#ifdef _MSC_VER
#pragma warning(disable : 4701)
#endif
auto RecurrentNeuron::output(const std::vector<float>& inputs, bool temporalReset) -> float
{
    if (temporalReset)
    {
        this->reset();
    }
    this->lastInputs.pushBack(inputs);
    this->previousSum = this->sum;
    this->previousOutput = this->lastOutput;
    this->sum = 0;
    size_t w = 0;
    float tmp = 0.0F;  // to activate the SIMD optimization
    assert(this->weights.size() == inputs.size() + 2);
#pragma omp simd
    for (w = 0; w < inputs.size(); ++w)
    {
        tmp += inputs[w] * this->weights[w];
    }
    this->sum = tmp + this->previousOutput * this->weights[w] + this->bias * this->weights[w + 1];
    const float output = outputFunction->function(sum);
    this->lastOutput = output;
    return output;
#ifdef _MSC_VER
#pragma warning(default : 4701)
#endif
}

auto RecurrentNeuron::backOutput(float error) -> std::vector<float>&
{
    error = error * this->outputFunction->derivative(this->sum);
    assert(this->weights.size() == this->errors.size() + 2);
#pragma omp simd  // seems to do nothing
    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    this->optimizer->updateWeights(*this, error);
    return this->errors;
}

void RecurrentNeuron::train(float error)
{
    error = error * this->outputFunction->derivative(this->sum);
    this->optimizer->updateWeights(*this, error);
}

inline void RecurrentNeuron::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}

auto RecurrentNeuron::isValid() const -> errorType
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 2)
    {
        return errorType::recurrentNeuronWrongNumberOfWeight;
    }
    return this->Neuron::isValid();
}

auto RecurrentNeuron::operator==(const RecurrentNeuron& neuron) const -> bool
{
    return this->Neuron::operator==(neuron) && this->lastOutput == neuron.lastOutput &&
           this->previousOutput == neuron.previousOutput && this->recurrentError == neuron.recurrentError &&
           this->previousSum == neuron.previousSum;
}

auto RecurrentNeuron::operator!=(const RecurrentNeuron& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal