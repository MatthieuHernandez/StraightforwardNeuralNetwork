#include "L1Regularization.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <functional>

namespace snn::internal
{
L1Regularization::L1Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer),
      value(value)
{
}

L1Regularization::L1Regularization(const L1Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer),
      value(regularization.value)
{
}

auto L1Regularization::clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer>
{
    return std::make_unique<L1Regularization>(*this, newLayer);
}

void L1Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfAbsNeuronWeights() * this->value;
    std::ranges::transform(inputErrors, inputErrors.begin(),
                           bind(std::plus<float>(), std::placeholders::_1, regularization));
}

auto L1Regularization::summary() const -> std::string
{
    std::stringstream summary;
    summary << "L1Regularization(" << value << ")\n";
    return summary.str();
}

auto L1Regularization::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const L1Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->value == o.value;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

auto L1Regularization::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
}  // namespace snn::internal
