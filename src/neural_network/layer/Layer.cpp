#include "Layer.hpp"
#include "neuron/SimpleNeuron.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GateRecurrentUnit.hpp"

using namespace snn;
using namespace internal;

extern template class Layer<SimpleNeuron>;
extern template class Layer<RecurrentNeuron>;
extern template class Layer<GateRecurrentUnit>;

BOOST_CLASS_EXPORT(Layer<SimpleNeuron>)
BOOST_CLASS_EXPORT(Layer<RecurrentNeuron>)
BOOST_CLASS_EXPORT(Layer<GateRecurrentUnit>)

