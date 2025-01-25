#include "NeuralNetworkOptimizer.hpp"

#include <boost/serialization/export.hpp>
#include <typeinfo>

using namespace std;
using namespace snn;
using namespace internal;

auto NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code();
}

auto NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    return !(*this == optimizer);
}
