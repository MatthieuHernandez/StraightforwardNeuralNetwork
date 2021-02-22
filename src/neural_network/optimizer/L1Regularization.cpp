#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "L1Regularization.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(L1Regularization)

L1Regularization::L1Regularization(const float value)
    : LayerOptimizer(), value(value)
{
}

unique_ptr<LayerOptimizer> L1Regularization::clone(LayerOptimizer* optimizer) const
{
    return make_unique<L1Regularization>(*this);
}

void L1Regularization::applyBefore(vector<float>& inputs)
{
    transform(inputs.begin(), inputs.end(), inputs.begin(), bind(multiplies<float>(), placeholders::_1, this->reverseValue));
}

void L1Regularization::applyAfterForBackpropagation(vector<float>& outputs)
{
    for (auto& o : outputs)
    {
        if (rand() / static_cast<float>(RAND_MAX) < value)
            o = 0.0f;
    }
}

bool L1Regularization::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const L1Regularization&>(optimizer);
        return this->value == o.value
            && this->reverseValue == o.reverseValue;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool L1Regularization::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
