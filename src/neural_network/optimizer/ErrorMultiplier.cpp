#include "ErrorMultiplier.hpp"

#include <sstream>

using namespace std;
using namespace snn;
using namespace internal;

ErrorMultiplier::ErrorMultiplier(float factor, BaseLayer* layer)
    : LayerOptimizer(layer), factor(factor)
{
}

ErrorMultiplier::ErrorMultiplier(const ErrorMultiplier& errorMultiplier, const BaseLayer* layer)
    : LayerOptimizer(layer)
{
    this->factor = errorMultiplier.factor;
}

std::unique_ptr<LayerOptimizer> ErrorMultiplier::clone(const BaseLayer* newLayer) const
{
    return make_unique<ErrorMultiplier>(*this, newLayer);
}

void ErrorMultiplier::applyBeforeBackpropagation(std::vector<float>& inputErrors)
{
    ranges::transform(inputErrors, inputErrors.begin(), bind(multiplies<float>(), placeholders::_1, this->factor));
}

std::string ErrorMultiplier::summary() const
{
    stringstream ss;
    ss << "ErrorMultiplier(" << factor << ")" << endl;
    return ss.str();
}

bool ErrorMultiplier::operator==(const LayerOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const ErrorMultiplier&>(optimizer);
        return this->LayerOptimizer::operator==(optimizer)
            && this->factor == o.factor;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool ErrorMultiplier::operator!=(const LayerOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
