#include <boost/serialization/export.hpp>
#include "StochasticGradientDescent.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(StochasticGradientDescent)

StochasticGradientDescent::StochasticGradientDescent(const float learningRate, const float momentum)
    : learningRate(learningRate), momentum(momentum)
{
}

shared_ptr<NeuralNetworkOptimizer> StochasticGradientDescent::clone() const
{
    return make_shared<StochasticGradientDescent>(*this);
}

inline
void StochasticGradientDescent::updateWeight(const float& error, float& weight, float& previousDeltaWeight, const float& lastInput) const
{
        auto deltaWeights = learningRate * error * lastInput;
        deltaWeights += momentum * previousDeltaWeight;
        weight += deltaWeights;
        previousDeltaWeight = deltaWeights;
}

int StochasticGradientDescent::isValid()
{
    if (this->learningRate <= 0.0f || this->learningRate >= 1.0f)
        return 103;
    if (this->momentum < 0.0f || this->momentum > 1.0f)
        return 104;
    return 0;
}

bool StochasticGradientDescent::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const StochasticGradientDescent&>(optimizer);
        return typeid(*this).hash_code() == typeid(optimizer).hash_code()
            && this->learningRate == o.learningRate
            && this->momentum == o.momentum;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool StochasticGradientDescent::operator!=(const NeuralNetworkOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
