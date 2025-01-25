#include "Softmax.hpp"

#include <algorithm>
#include <boost/serialization/export.hpp>
#include <numeric>

#include "BaseLayer.hpp"
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Softmax::Softmax(const BaseLayer* layer)
    : LayerOptimizer(layer)
{
}

Softmax::Softmax([[maybe_unused]] const Softmax& softmax, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
}

auto Softmax::clone(const BaseLayer* newLayer) const -> unique_ptr<LayerOptimizer>
{
    return make_unique<Softmax>(*this, newLayer);
}

inline void Softmax::computeSoftmax(std::vector<float>& outputs)
{
    // max can be replace by a const (e.g. 10)
    const auto max = ranges::max(outputs);
    const auto sumExp = accumulate(outputs.begin(), outputs.end(), 0.0F,
                                   [max](float sumExp, float& value)
                                   {
                                       value -= max;
                                       return sumExp + exp(value);
                                   });
    for (auto& output : outputs)
    {
        const auto value = exp(output) / sumExp;
        if (fpclassify(value) != FP_NORMAL && fpclassify(value) != FP_ZERO)
            output = 0;
        else
            output = value;
    }
}

inline void Softmax::applyAfterOutputForTraining(std::vector<float>& outputs, [[maybe_unused]] bool temporalReset)
{
    computeSoftmax(outputs);
}

inline void Softmax::applyAfterOutputForTesting(std::vector<float>& outputs) { computeSoftmax(outputs); }

inline void Softmax::applyBeforeBackpropagation([[maybe_unused]] std::vector<float>& inputErrors) {}

auto Softmax::summary() const -> std::string
{
    stringstream ss;
    ss << "Softmax";
    return ss.str();
}

auto Softmax::operator==(const LayerOptimizer& optimizer) const -> bool
{
    try
    {
        dynamic_cast<const Softmax&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer);
    }
    catch (bad_cast&)
    {
        return false;
    }
}

auto Softmax::operator!=(const LayerOptimizer& optimizer) const -> bool { return !(*this == optimizer); }
