#include <boost/serialization/export.hpp>
#include "LayerOptimizer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LayerOptimizer)

LayerOptimizer::LayerOptimizer(BaseLayer* layer)
    : layer(layer)
{
}
