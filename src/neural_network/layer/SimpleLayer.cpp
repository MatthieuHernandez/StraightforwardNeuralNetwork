#include "SimpleLayer.hpp"
#include "neuron/SimpleNeuron.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

using namespace snn;
using namespace internal;

extern template class internal::SimpleLayer<SimpleNeuron>; // must use a nested-name-specifier for GCC compiler
extern template class internal::SimpleLayer<RecurrentNeuron>;
extern template class internal::SimpleLayer<GatedRecurrentUnit>;

BOOST_CLASS_EXPORT(SimpleLayer<SimpleNeuron>)
BOOST_CLASS_EXPORT(SimpleLayer<RecurrentNeuron>)
BOOST_CLASS_EXPORT(SimpleLayer<GatedRecurrentUnit>)

template <>
std::vector<float> SimpleLayer<RecurrentNeuron>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}

template <>
std::vector<float> SimpleLayer<GatedRecurrentUnit>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}
