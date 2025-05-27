#include "SimpleNeuron.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

namespace snn::internal
{
SimpleNeuron::SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Neuron(model, std::move(optimizer))
{
}

auto SimpleNeuron::output(const std::vector<float>& inputs) -> float
{
    this->lastInputs.pushBack(inputs);
    float tmp = 0.0F;  // to activate the SIMD optimization
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

auto SimpleNeuron::backOutput(float error) -> std::vector<float>&
{
    error *= this->outputFunction->derivative(this->sum);
    this->lastErrors.pushBack(error);
    assert(this->weights.size() == this->errors.size() + 1);
#pragma omp simd  // seems to do nothing
    for (size_t w = 0; w < this->errors.size(); ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    return this->errors;
}

void SimpleNeuron::back(float error)
{
    error *= this->outputFunction->derivative(this->sum);
    this->lastErrors.pushBack(error);
}

void SimpleNeuron::train() { this->optimizer->updateWeights(*this); }

auto SimpleNeuron::isValid() const -> errorType { return this->Neuron::isValid(); }

auto SimpleNeuron::operator==(const SimpleNeuron& neuron) const -> bool { return this->Neuron::operator==(neuron); }

auto SimpleNeuron::operator!=(const SimpleNeuron& neuron) const -> bool { return !(*this == neuron); }
}  // namespace snn::internal
