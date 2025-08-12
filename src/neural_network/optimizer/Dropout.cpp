#include "Dropout.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <functional>

#include "BaseLayer.hpp"
#include "Tools.hpp"

namespace snn::internal
{
Dropout::Dropout(const float value, const BaseLayer* layer)
    : LayerOptimizer(layer),
      value(value),
      reverseValue(1.0F - this->value)
{
    auto size = layer->getNumberOfNeurons();
    this->presenceProbabilities.resize(size);
    this->dist = std::uniform_real_distribution<>(0.0, 1.0);
    // NOLINTNEXTLINE(modernize-use-ranges):  std::ranges::generate doesn't support std::vector<bool>.
    std::generate(this->presenceProbabilities.begin(), this->presenceProbabilities.end(),
                  [&] { return dist(tools::Rng()) >= this->value; });
}

Dropout::Dropout(const Dropout& dropout, const BaseLayer* layer)
    : LayerOptimizer(layer),
      value(dropout.value),
      reverseValue(dropout.reverseValue),
      presenceProbabilities(dropout.presenceProbabilities)
{
}

auto Dropout::clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer>
{
    return std::make_unique<Dropout>(*this, newLayer);
}

void Dropout::applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset)
{
    if (temporalReset)
    {
        // NOLINTNEXTLINE(modernize-use-ranges):  std::ranges::generate doesn't support std::vector<bool>.
        std::generate(this->presenceProbabilities.begin(), this->presenceProbabilities.end(),
                      [&] { return dist(tools::Rng()) >= this->value; });
    }
    std::ranges::transform(outputs, this->presenceProbabilities, outputs.begin(), std::multiplies<float>());
}

void Dropout::applyAfterOutputForTesting(std::vector<float>& outputs)
{
    std::ranges::transform(outputs, outputs.begin(), [this](auto&& value) { return value * this->reverseValue; });
}

void Dropout::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    std::ranges::transform(inputErrors, this->presenceProbabilities, inputErrors.begin(), std::multiplies<float>());
}

auto Dropout::summary() const -> std::string
{
    std::stringstream summary;
    summary << "Dropout(" << value << ")\n";
    return summary.str();
}

auto Dropout::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const Dropout&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer) && this->value == o.value &&
               this->reverseValue == o.reverseValue;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}
}  // namespace snn::internal
