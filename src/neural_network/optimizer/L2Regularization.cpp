#include "L2Regularization.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <functional>

namespace snn::internal
{
L2Regularization::L2Regularization(const float value, BaseLayer* layer)
    : LayerOptimizer(layer),
      value(value)
{
}

L2Regularization::L2Regularization(const L2Regularization& regularization, const BaseLayer* layer)
    : LayerOptimizer(layer),
      value(regularization.value)
{
}

auto L2Regularization::clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer>
{
    return std::make_unique<L2Regularization>(*this, newLayer);
}

void L2Regularization::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    auto regularization = this->layer->getAverageOfSquareNeuronWeights() * this->value;
    std::ranges::transform(inputErrors, inputErrors.begin(),
                           [regularization](auto&& value) { return value + regularization; });
}

auto L2Regularization::summary() const -> std::string
{
    std::stringstream summary;
    summary << "L2Regularization(" << value << ")\n";
    return summary.str();
}

auto L2Regularization::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const L2Regularization&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->value == o.value;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

auto L2Regularization::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
}  // namespace snn::internal