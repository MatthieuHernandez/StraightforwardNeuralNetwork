#include <typeinfo>
#include <boost/serialization/export.hpp>
#include "Optimizer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Optimizer)

bool Optimizer::operator==(const Optimizer& optimizer) const
{
    return typeid(*this).hash_code() == typeid(optimizer).hash_code();
}

bool Optimizer::operator!=(const Optimizer& optimizer) const
{
    return !(*this == optimizer);
}
