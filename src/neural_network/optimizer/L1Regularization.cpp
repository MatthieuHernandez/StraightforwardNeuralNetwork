#include "L1Regularization.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <functional>

using namespace std;
using namespace snn;
using namespace internal;

L1Regularization::L1Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer),
      value(value)
{
}

L1Regularization::L1Regularization(const L1Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->value = regularization.value;
}

auto L1Regularization::clone(const BaseLayer* newLayer) const -> unique_ptr<LayerOptimizer>
{
    return make_unique<L1Regularization>(*this, newLayer);
}

void L1Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfAbsNeuronWeights() * this->value;
    ranges::transform(inputErrors, inputErrors.begin(), bind(plus<float>(), placeholders::_1, regularization));
}

auto L1Regularization::summary() const -> std::string
{
    stringstream ss;
    ss << "L1Regularization(" << value << ")" << endl;
    return ss.str();
}

auto L1Regularization::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const L1Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->value == o.value;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

auto L1Regularization::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
