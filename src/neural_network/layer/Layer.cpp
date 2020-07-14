#include "Layer.hpp"
#include "neuron/RecurrentNeuron.hpp"

using namespace snn;
using namespace snn::internal;

extern template class Layer<Neuron>;
extern template class Layer<RecurrentNeuron>;

BOOST_CLASS_EXPORT(Layer<Neuron>)
BOOST_CLASS_EXPORT(Layer<RecurrentNeuron>)

