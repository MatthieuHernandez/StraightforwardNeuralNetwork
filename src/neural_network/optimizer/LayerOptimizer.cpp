#include "LayerOptimizer.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

LayerOptimizer::LayerOptimizer(const BaseLayer* layer)
    : layer(layer)
{
}

auto LayerOptimizer::operator==(const LayerOptimizer& optimizer) const -> bool
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code() &&
           typeid(this->layer).hash_code() == typeid(optimizer.layer).hash_code();
}

auto LayerOptimizer::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
