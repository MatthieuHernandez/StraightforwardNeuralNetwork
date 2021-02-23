#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "Dropout.hpp"
#include "../layer/BaseLayer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Dropout)

auto randomProbability = [](const float p) -> auto { return static_cast<bool>(rand() / static_cast<float>(RAND_MAX) >= p); };

Dropout::Dropout(const float value, BaseLayer* layer)
    : LayerOptimizer(layer), value(value)
{
    this->reverseValue = 1.0f - this->value;
    auto size = static_cast<BaseLayer*>(layer)->getNumberOfNeurons();
    this->presenceProbabilities.resize(size);
    std::generate(this->presenceProbabilities.begin(), this->presenceProbabilities.end(), randomProbability(this->value));
}

unique_ptr<LayerOptimizer> Dropout::clone(LayerOptimizer* optimizer) const
{
    return make_unique<Dropout>(*this);
}

void Dropout::applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset)
{
    if(temporalReset)
         std::generate(this->presenceProbabilities.begin(), this->presenceProbabilities.end(), randomProbability(this->value));
    transform(outputs.begin(), outputs.end(), this->presenceProbabilities.begin(), multiplies<float>());
}

void Dropout::applyAfterOutputForTesting(std::vector<float>& outputs)
{
    transform(outputs.begin(), outputs.end(), outputs.begin(), bind(multiplies<float>(), placeholders::_1, this->reverseValue));
}

void Dropout::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    transform(inputErrors.begin(), inputErrors.end(), this->presenceProbabilities.begin(), multiplies<float>());
}

bool Dropout::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const Dropout&>(optimizer);
        return this->value == o.value
            && this->reverseValue == o.reverseValue;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool Dropout::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
