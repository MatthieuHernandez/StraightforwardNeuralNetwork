#include <boost/serialization/export.hpp>
#include "LayerOptimizer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LayerOptimizer)

bool LayerOptimizer::operator==(const Optimizer& optimizer) const
{
    return this->Optimizer::operator==(optimizer);
}

bool LayerOptimizer::operator!=(const Optimizer& optimizer) const
{
    return this->Optimizer::operator!=(optimizer);
}
