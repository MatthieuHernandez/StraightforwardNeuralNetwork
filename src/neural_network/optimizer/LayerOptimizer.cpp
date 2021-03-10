#include <boost/serialization/export.hpp>
#include "LayerOptimizer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LayerOptimizer)

LayerOptimizer::LayerOptimizer(const BaseLayer* layer)
    : layer(layer)
{
}

bool LayerOptimizer::operator==(const LayerOptimizer& optimizer) const
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code()
        && typeid(this->layer).hash_code() == typeid(optimizer.layer).hash_code();
}

bool LayerOptimizer::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
