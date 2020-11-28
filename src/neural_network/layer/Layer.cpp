#include <boost/serialization/export.hpp>
#include "Layer.hpp"
#include "neuron/SimpleNeuron.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

using namespace snn;
using namespace internal;

extern template class internal::Layer<SimpleNeuron>;
extern template class internal::Layer<RecurrentNeuron>;
extern template class internal::Layer<GatedRecurrentUnit>;


BOOST_CLASS_EXPORT(Layer<SimpleNeuron>)
BOOST_CLASS_EXPORT(Layer<RecurrentNeuron>)
BOOST_CLASS_EXPORT(Layer<GatedRecurrentUnit>)

