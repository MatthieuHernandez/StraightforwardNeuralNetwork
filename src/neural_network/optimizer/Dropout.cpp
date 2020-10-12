#include "Dropout.hpp"
#include <algorithm>
#include <functional>

using namespace std;
using namespace snn;
using namespace internal;

Dropout::Dropout(const float value)
    : LayerOptimizer(), value(value)
{
    this->reverseValue = 1.0f / (1.0f - this->value);
}

std::unique_ptr<LayerOptimizer> Dropout::clone(LayerOptimizer* optimizer) const
{
    return make_unique<Dropout>(*this);
}

void Dropout::apply(std::vector<float>& output)
{
    transform(output.begin(), output.end(), output.begin(),
              std::bind(std::multiplies<float>(), std::placeholders::_1, this->reverseValue));
}

void Dropout::applyForBackpropagation(std::vector<float>& output)
{
    for (auto& o : output)
    {
        if (rand() / static_cast<float>(RAND_MAX) < value)
            o = 0.0f;
    }
}

bool Dropout::operator==(const Optimizer& optimizer) const
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

bool Dropout::operator!=(const Optimizer& optimizer) const
{
    return !(*this == optimizer);
}
