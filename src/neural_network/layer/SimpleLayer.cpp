#include "SimpleLayer.hpp"

#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
template class SimpleLayer<SimpleNeuron>;
template class SimpleLayer<RecurrentNeuron>;
template class SimpleLayer<GatedRecurrentUnit>;

template <>
auto SimpleLayer<RecurrentNeuron>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}

template <>
auto SimpleLayer<GatedRecurrentUnit>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}
}  // namespace snn::internal