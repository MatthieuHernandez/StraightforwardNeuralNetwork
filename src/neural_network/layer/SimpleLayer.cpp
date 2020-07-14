#include "SimpleLayer.hpp"
#include "neuron/RecurrentNeuron.hpp"

using namespace snn;
using namespace snn::internal;

extern template class SimpleLayer<Neuron>;
extern template class SimpleLayer<RecurrentNeuron>;

BOOST_CLASS_EXPORT(SimpleLayer<Neuron>)
BOOST_CLASS_EXPORT(SimpleLayer<RecurrentNeuron>)

