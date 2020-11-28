#include <boost/serialization/export.hpp>
#include "StochasticGradientDescent.hpp"
#include "../layer/neuron/RecurrentNeuron.hpp"
#include "../layer/neuron/SimpleNeuron.hpp"

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

void StochasticGradientDescent::updateWeights(SimpleNeuron& neuron, float error) const
{
    for (size_t w = 0; w < neuron.weights.size(); ++w)
    {
        auto deltaWeights = this->learningRate * error * neuron.lastInputs[w];
        deltaWeights += this->momentum * neuron.previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights;
        neuron.previousDeltaWeights[w] = deltaWeights;
    }
}

void StochasticGradientDescent::updateWeights(RecurrentNeuron& neuron, float error) const
{
    size_t w;
    for (w = 0; w < neuron.lastInputs.size(); ++w)
    {
        auto deltaWeights = this->learningRate * error * neuron.lastInputs[w];
        deltaWeights += this->momentum * neuron.previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights;
        neuron.previousDeltaWeights[w] = deltaWeights;
    }
    neuron.recurrentError = error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) *
        neuron.weights[w];

    auto deltaWeights = this->learningRate * neuron.recurrentError * neuron.previousOutput;
    deltaWeights += this->momentum * neuron.previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights;
    neuron.previousDeltaWeights[w] = deltaWeights;
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
        return this->NeuralNetworkOptimizer::operator==(optimizer)
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
