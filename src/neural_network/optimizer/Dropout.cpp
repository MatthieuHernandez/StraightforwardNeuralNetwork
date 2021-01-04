#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "Dropout.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Dropout)

Dropout::Dropout(const float value)
    : LayerOptimizer(), value(value)
{
    this->reverseValue = 1.0f - this->value;
}

unique_ptr<LayerOptimizer> Dropout::clone(LayerOptimizer* optimizer) const
{
    return make_unique<Dropout>(*this);
}

void Dropout::applyBefore(vector<float>& inputs)
{
    transform(inputs.begin(), inputs.end(), inputs.begin(), bind(multiplies<float>(), placeholders::_1, this->reverseValue));
}

void Dropout::applyAfterForBackpropagation(vector<float>& outputs)
{
    for (auto& o : outputs)
    {
        if (rand() / static_cast<float>(RAND_MAX) < value)
            o = 0.0f;
    }
}

bool Dropout::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const Dropout&>(optimizer);
        return this->value == o.value
            && this->reverseValue == o.reverseValue;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool Dropout::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
