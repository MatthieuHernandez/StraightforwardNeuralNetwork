#include "SimpleLayer.hpp"
#include "neuron/RecurrentNeuron.hpp"

using namespace snn;
using namespace internal;

extern template class SimpleLayer<Neuron>;
extern template class SimpleLayer<RecurrentNeuron>;

BOOST_CLASS_EXPORT(SimpleLayer<Neuron>)
BOOST_CLASS_EXPORT(SimpleLayer<RecurrentNeuron>)

template <>
std::vector<float> SimpleLayer<RecurrentNeuron>::output(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs, temporalReset);
    }
    return outputs;
}