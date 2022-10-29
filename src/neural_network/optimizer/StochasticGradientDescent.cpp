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

void StochasticGradientDescent::updateWeights(SimpleNeuron& neuron, const float error) const
{
    auto lr = this->learningRate / neuron.lastInputs.size(); // to activate the SIMD optimization
    auto m = this->momentum;
    #pragma omp simd
    for (size_t w = 0; w < neuron.weights.size(); ++w)
    {
        const auto deltaWeights = lr * error * neuron.lastInputs.back()[w] + m * neuron.previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights;
        neuron.previousDeltaWeights[w] = deltaWeights;
    }
    neuron.bias += lr * error * neuron.bias;
}

#ifdef _MSC_VER
#pragma warning(disable:4701)
#endif
void StochasticGradientDescent::updateWeights(RecurrentNeuron& neuron, float error) const
{
    size_t w = 0;
    auto lr = this->learningRate / neuron.lastInputs.size(); // to activate the SIMD optimization
    auto m = this->momentum; 
    #pragma omp simd
    for (w = 0; w < neuron.lastInputs.back().size(); ++w)
    {
        const auto deltaWeights = lr * error * neuron.lastInputs.back()[w] + m * neuron.previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights;
        neuron.previousDeltaWeights[w] = deltaWeights;
    }
    neuron.bias += lr * error * neuron.bias;
    neuron.recurrentError = error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) * neuron.weights[w];

    auto deltaWeights = lr * neuron.recurrentError * neuron.previousOutput + m * neuron.previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights;
    neuron.previousDeltaWeights[w] = deltaWeights;
    #ifdef _MSC_VER
    #pragma warning(default:4701)
    #endif
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
