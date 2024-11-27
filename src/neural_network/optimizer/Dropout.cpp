#include <algorithm>
#include <functional>
#include <boost/serialization/export.hpp>
#include "Dropout.hpp"
#include "BaseLayer.hpp"
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Dropout::Dropout(const float value, const BaseLayer* layer)
    : LayerOptimizer(layer), value(value)
{
    this->reverseValue = 1.0f - this->value;
    auto size = layer->getNumberOfNeurons();
    this->presenceProbabilities.resize(size);
    this->dist = uniform_real_distribution<>(0.0, 1.0);
    std::generate(this->presenceProbabilities.begin(),
                  this->presenceProbabilities.end(),
                  [&] { return dist(tools::rng) >= this->value; });
}

Dropout::Dropout(const Dropout& dropout, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->value = dropout.value;
    this->reverseValue = dropout.reverseValue;
    this->presenceProbabilities = dropout.presenceProbabilities;
}

unique_ptr<LayerOptimizer> Dropout::clone(const BaseLayer* newLayer) const
{
    return make_unique<Dropout>(*this, newLayer);
}

void Dropout::applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset)
{
    if (temporalReset)
        std::generate(this->presenceProbabilities.begin(),
                      this->presenceProbabilities.end(),
                      [&] { return dist(tools::rng) >= this->value; });
    ranges::transform(outputs, this->presenceProbabilities, outputs.begin(), multiplies<float>());
}

void Dropout::applyAfterOutputForTesting(std::vector<float>& outputs)
{
    ranges::transform(outputs, outputs.begin(), bind(multiplies<float>(), placeholders::_1, this->reverseValue));
}

void Dropout::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    ranges::transform(inputErrors, this->presenceProbabilities, inputErrors.begin(), multiplies<float>());
}

std::string Dropout::summary() const
{
    stringstream ss;
    ss << "Dropout(" << value << ")" << endl;
    return ss.str();
}

bool Dropout::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const Dropout&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer)
            && this->value == o.value
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
