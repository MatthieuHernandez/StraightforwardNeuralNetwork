#include "L2Regularization.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <functional>

using namespace std;
using namespace snn;
using namespace internal;

L2Regularization::L2Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer),
      value(value)
{
}

L2Regularization::L2Regularization(const L2Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->value = regularization.value;
}

auto L2Regularization::clone(const BaseLayer* newLayer) const -> unique_ptr<LayerOptimizer>
{
    return make_unique<L2Regularization>(*this, newLayer);
}

void L2Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfSquareNeuronWeights() * this->value;
    ranges::transform(inputErrors, inputErrors.begin(), bind(plus<float>(), placeholders::_1, regularization));
}

auto L2Regularization::summary() const -> std::string
{
    stringstream ss;
    ss << "L2Regularization(" << value << ")" << endl;
    return ss.str();
}

auto L2Regularization::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const L2Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->value == o.value;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

auto L2Regularization::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
