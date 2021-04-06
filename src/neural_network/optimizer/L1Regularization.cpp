#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "L1Regularization.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(L1Regularization)

L1Regularization::L1Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer), value(value)
{
}

L1Regularization::L1Regularization(const L1Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->value = regularization.value;
}

unique_ptr<LayerOptimizer> L1Regularization::clone(const BaseLayer* newLayer) const
{
    return make_unique<L1Regularization>(*this, newLayer);
}

void L1Regularization::applyAfterOutputForTraining(std::vector<float>& outputs, bool)
{
}

void L1Regularization::applyAfterOutputForTesting(std::vector<float>& outputs)
{
}

void L1Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfAbsNeuronWeights() * this->value;
    ranges::transform(inputErrors, inputErrors.begin(), bind(plus<float>(), placeholders::_1, regularization));
}

bool L1Regularization::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const L1Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer)
            && this->value == o.value;
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
