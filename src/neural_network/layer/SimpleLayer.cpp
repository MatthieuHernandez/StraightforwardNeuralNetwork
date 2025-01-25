#include "SimpleLayer.hpp"

#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

using namespace snn;
using namespace internal;

extern template class internal::SimpleLayer<SimpleNeuron>;  // must use a nested-name-specifier for GCC compiler
extern template class internal::SimpleLayer<RecurrentNeuron>;
extern template class internal::SimpleLayer<GatedRecurrentUnit>;

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
