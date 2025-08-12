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
    float sum = 0.0F;  // to activate the SIMD optimization
    assert(this->weights.size() == inputs.size() + 1);
    size_t w = 0;
#pragma omp simd
    for (w = 0; w < inputs.size(); ++w)
    {
        sum += inputs[w] * this->weights[w];
    }
    sum += this->weights[w] * this->bias;
    this->lastSum.pushBack(sum);
    return this->outputFunction->function(sum);
}

auto SimpleNeuron::backOutput(float error) -> std::vector<float>&
{
    const auto& sum = *this->lastSum.getBack();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
    assert(this->weights.size() == this->errors.size() + 1);
#pragma omp simd  // seems to do nothing
    for (size_t w = 0; w < this->errors.size(); ++w)
    {
        this->errors[w] = e * this->weights[w];
    }
    return this->errors;
}

void SimpleNeuron::back(float error)
{
    const auto& sum = *this->lastSum.getBack();
    const auto e = error * this->outputFunction->derivative(sum);
    this->lastError.pushBack(e);
}

void SimpleNeuron::train() { this->optimizer->updateWeights(*this); }

auto SimpleNeuron::isValid() const -> errorType { return this->Neuron::isValid(); }

auto SimpleNeuron::operator==(const SimpleNeuron& neuron) const -> bool { return this->Neuron::operator==(neuron); }
}  // namespace snn::internal
