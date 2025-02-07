#include "ErrorMultiplier.hpp"

#include <algorithm>
#include <sstream>

namespace snn::internal
{
ErrorMultiplier::ErrorMultiplier(float factor, BaseLayer* layer)
    : LayerOptimizer(layer),
      factor(factor)
{
}

ErrorMultiplier::ErrorMultiplier(const ErrorMultiplier& errorMultiplier, const BaseLayer* layer)
    : LayerOptimizer(layer),
      factor(errorMultiplier.factor)
{
}

auto ErrorMultiplier::clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer>
{
    return std::make_unique<ErrorMultiplier>(*this, newLayer);
}

void ErrorMultiplier::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    std::ranges::transform(inputErrors, inputErrors.begin(),
                           bind(std::multiplies<float>(), std::placeholders::_1, this->factor));
}

auto ErrorMultiplier::summary() const -> std::string
{
    std::stringstream summary;
    summary << "ErrorMultiplier(" << factor << ")\n";
    return summary.str();
}

auto ErrorMultiplier::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const ErrorMultiplier&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->factor == o.factor;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

auto ErrorMultiplier::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
}  // namespace snn::internal
