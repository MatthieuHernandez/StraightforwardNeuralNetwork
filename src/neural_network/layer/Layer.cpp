#include <boost/serialization/export.hpp>
#include "Layer.hpp"
#include "neuron/SimpleNeuron.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

using namespace snn;
using namespace internal;

extern template class internal::Layer<SimpleNeuron>; // must use a nested-name-specifier for GCC compiler
extern template class internal::Layer<RecurrentNeuron>;
extern template class internal::Layer<GatedRecurrentUnit>;
