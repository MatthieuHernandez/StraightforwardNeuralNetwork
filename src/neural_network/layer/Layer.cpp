#include "Layer.hpp"

#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
template class Layer<SimpleNeuron>;
template class Layer<RecurrentNeuron>;
template class Layer<GatedRecurrentUnit>;
}  // namespace snn::internal
