#include "Layer.hpp"

#include <boost/serialization/export.hpp>

#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
extern template class Layer<SimpleNeuron>;
extern template class Layer<RecurrentNeuron>;
extern template class Layer<GatedRecurrentUnit>;
}  // namespace snn::internal
