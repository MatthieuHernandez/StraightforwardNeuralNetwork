#include "Layer.hpp"

#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
template class Layer<SimpleNeuron>;
template class Layer<RecurrentNeuron>;
template class Layer<GatedRecurrentUnit>;
static_assert(LearningObject<Layer<SimpleNeuron>>);
static_assert(LearningObject<Layer<RecurrentNeuron>>);
static_assert(LearningObject<Layer<GatedRecurrentUnit>>);
}  // namespace snn::internal
