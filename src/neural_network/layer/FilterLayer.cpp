#include <boost/serialization/export.hpp>
#include "FilterLayer.hpp"
#include "neuron/input/NeuronInputFromConvolution2D.hpp"

using namespace std;
using namespace internal;

extern template class internal::FilterLayer<vector<float>>; // must use a nested-name-specifier for GCC compiler
extern template class internal::FilterLayer<NeuronInputFromConvolution2D>;

BOOST_CLASS_EXPORT(FilterLayer<vector<float>>)
BOOST_CLASS_EXPORT(FilterLayer<NeuronInputFromConvolution2D>)
