#include "NeuralNetworkOptimizer.hpp"

#include <boost/serialization/export.hpp>

namespace snn::internal
{
auto NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code();
}

auto NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    return !(*this == optimizer);
}
}  // namespace snn::internal
