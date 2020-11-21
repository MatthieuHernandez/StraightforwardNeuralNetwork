#include <typeinfo>
#include <boost/serialization/export.hpp>
#include "NeuralNetworkOptimizer.hpp"

using namespace std;
using namespace snn;

//BOOST_CLASS_EXPORT(NeuralNetworkOptimizer)

bool NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    auto a = typeid(*this).hash_code();
    auto b = typeid(optimizer).hash_code();
    return a == b;

}

bool NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
