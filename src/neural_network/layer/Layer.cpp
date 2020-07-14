#include "Layer.hpp"
#include "perceptron/RecurrentNeuron.hpp"

using namespace snn;
using namespace snn::internal;

extern template class Layer<Perceptron>;
extern template class Layer<RecurrentNeuron>;

BOOST_CLASS_EXPORT(Layer<Perceptron>)
BOOST_CLASS_EXPORT(Layer<RecurrentNeuron>)

