#include <typeinfo>
#include <boost/serialization/export.hpp>
#include "NeuralNetworkOptimizer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

//BOOST_CLASS_EXPORT(NeuralNetworkOptimizer)

void NeuralNetworkOptimizer::operator++()
{
    if (this->t < 100000)
        this->t++;
}

bool NeuralNetworkOptimizer::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code()
        && this->t == optimizer.t;

}

bool NeuralNetworkOptimizer::operator!=(const NeuralNetworkOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
