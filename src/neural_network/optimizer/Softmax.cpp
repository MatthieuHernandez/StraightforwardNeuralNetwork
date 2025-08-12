#include "Softmax.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <cmath>
#include <numeric>

#include "BaseLayer.hpp"

namespace snn::internal
{
Softmax::Softmax(const BaseLayer* layer)
    : LayerOptimizer(layer)
{
}

Softmax::Softmax([[maybe_unused]] const Softmax& softmax, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
}

auto Softmax::clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer>
{
    return std::make_unique<Softmax>(*this, newLayer);
}

inline void Softmax::computeSoftmax(std::vector<float>& outputs)
{
    // max can be replace by a const (e.g. 10)
    const auto max = std::ranges::max(outputs);
    const auto sumExp = std::accumulate(outputs.begin(), outputs.end(), 0.0F,
                                        [max](float sumExp, float& value)
                                        {
                                            value -= max;
                                            return sumExp + expf(value);
                                        });
    for (auto& output : outputs)
    {
        const auto value = std::expf(output) / sumExp;
        if (std::fpclassify(value) != FP_NORMAL && std::fpclassify(value) != FP_ZERO)
        {
            output = 0;
        }
        else
        {
            output = value;
        }
    }
}

inline void Softmax::applyAfterOutputForTraining(std::vector<float>& outputs, [[maybe_unused]] bool temporalReset)
{
    computeSoftmax(outputs);
}

inline void Softmax::applyAfterOutputForTesting(std::vector<float>& outputs) { computeSoftmax(outputs); }

inline void Softmax::applyBeforeBackpropagation([[maybe_unused]] std::vector<float>& inputErrors) {}

auto Softmax::summary() const -> std::string { return "Softmax"; }

auto Softmax::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        [[maybe_unused]] auto opti = dynamic_cast<const Softmax&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer);
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}
}  // namespace snn::internal
