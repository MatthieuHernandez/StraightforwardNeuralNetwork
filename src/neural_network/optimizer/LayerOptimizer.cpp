#include "LayerOptimizer.hpp"

#include <boost/serialization/export.hpp>

namespace snn::internal
{
LayerOptimizer::LayerOptimizer(const BaseLayer* layer)
    : layer(layer)
{
}

auto LayerOptimizer::operator==(const LayerOptimizer& optimizer) const -> bool
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code() &&
           typeid(this->layer).hash_code() == typeid(optimizer.layer).hash_code();
}
}  // namespace snn::internal
