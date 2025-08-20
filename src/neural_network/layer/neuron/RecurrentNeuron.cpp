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
    this->previousOutput = this->lastOutput;
    float sum = 0.0F;  // to activate the SIMD optimization
    size_t w = 0;
    assert(this->weights.size() == inputs.size() + 2);
#pragma omp simd
    for (w = 0; w < inputs.size(); ++w)
    {
        sum += inputs[w] * this->weights[w];
    }
    sum += this->previousOutput * this->weights[w] + this->bias * this->weights[w + 1];
    this->lastSum.pushBack(sum);
    const float output = outputFunction->function(sum);
    this->lastOutput = output;
    return output;
#ifdef _MSC_VER
#pragma warning(default : 4701)
#endif
}

auto RecurrentNeuron::backOutput(float error) -> std::vector<float>&
{
    const auto& sum = *this->lastSum.getBack();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
    assert(this->weights.size() == this->errors.size() + 2);
#pragma omp simd  // seems to do nothing
    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = e * this->weights[w];
    }
    return this->errors;
}

void RecurrentNeuron::back(float error)
{
    const auto& sum = *this->lastSum.getBack();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
}

void RecurrentNeuron::train() { this->optimizer->updateWeights(*this); }

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

void RecurrentNeuron::resetLearningVariables(int batchSize)
{
    if (batchSize != 1)
    {
        throw std::invalid_argument("The batch size should be 1 for reccurent neurons.");
    }
    this->Neuron::resetLearningVariables(batchSize);
    this->lastOutput = 0;
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}
}  // namespace snn::internal
