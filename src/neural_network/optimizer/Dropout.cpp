#include "Dropout.hpp"
#include <algorithm>
#include <functional>

using namespace std;
using namespace snn;
using namespace internal;

Dropout::Dropout(const float value)
    : value(value)
{
    reverseValue = 1.0f / (1.0f - this->value);
}

void Dropout::apply(std::vector<float>& output)
{
    transform(output.begin(), output.end(), output.begin(), std::bind(std::multiplies<float>(), std::placeholders::_1, 3));
}

void Dropout::applyForBackpropagation(std::vector<float>& output)
{
    for (auto& o : output)
    {
        if (rand() / static_cast<float>(RAND_MAX) < value)
            o = 0.0f;
    }
}

bool Dropout::operator==(const Dropout& d) const
{
    return this->value == d.value
        && this->reverseValue == d.reverseValue;
}

bool Dropout::operator!=(const Dropout& d) const
{
    return !(*this == d);
}
