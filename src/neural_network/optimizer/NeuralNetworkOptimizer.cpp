#include "NeuralNetworkOptimizer.hpp"

#include <boost/serialization/export.hpp>
#include <typeinfo>

using namespace std;
using namespace snn;
using namespace internal;

bool NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code();
}

bool NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const { return !(*this == optimizer); }
