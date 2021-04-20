#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "L2Regularization.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(L2Regularization)

L2Regularization::L2Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer), value(value)
{
}

L2Regularization::L2Regularization(const L2Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->value = regularization.value;
}

unique_ptr<LayerOptimizer> L2Regularization::clone(const BaseLayer* newLayer) const
{
    return make_unique<L2Regularization>(*this, newLayer);
}

void L2Regularization::applyAfterOutputForTraining([[maybe_unused]] std::vector<float>& outputs, bool)
{
}

void L2Regularization::applyAfterOutputForTesting([[maybe_unused]] std::vector<float>& outputs)
{
}

void L2Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfSquareNeuronWeights() * this->value;
    ranges::transform(inputErrors, inputErrors.begin(), bind(plus<float>(), placeholders::_1, regularization));
}

bool L2Regularization::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const L2Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer)
            && this->value == o.value;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool L2Regularization::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
