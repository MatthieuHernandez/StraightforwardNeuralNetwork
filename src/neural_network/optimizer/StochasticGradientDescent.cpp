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
    auto w = 0;
    const auto lr = this->learningRate / neuron.batchSize; // to activate the SIMD optimization
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto lastInputs = *neuron.lastInputs.getBack();
    const auto previousDeltaWeights = *neuron.previousDeltaWeights.getBack();
    vector<float> deltaWeights(neuron.weights.size());
    const auto lr_error = lr * error;
    #pragma omp simd
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr_error * lastInputs[w] + m * previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights[w];
    }
    deltaWeights[w] = lr * error * neuron.bias + m * previousDeltaWeights[w];
    neuron.weights[w] = deltaWeights[w];
    neuron.previousDeltaWeights.pushBack(deltaWeights);
}

#ifdef _MSC_VER
#pragma warning(disable:4701)
#endif
void StochasticGradientDescent::updateWeights(RecurrentNeuron& neuron, float error) const
{
    auto w = 0;
    const auto lr = this->learningRate / neuron.batchSize;
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto lastInputs = *neuron.lastInputs.getBack();
    const auto previousDeltaWeights = *neuron.previousDeltaWeights.getBack();
    vector<float> deltaWeights(neuron.weights.size());
    #pragma omp simd // info C5002: Omp simd loop not vectorized due to reason '1305' (Not enough type information.)
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr * error * lastInputs[w] + m * previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights[w];
    }
    deltaWeights[w] = lr * error * neuron.bias + m * previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights[w];
    neuron.recurrentError = error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) * neuron.weights[w];

    deltaWeights[w] = lr * neuron.recurrentError * neuron.previousOutput + m * previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights[w];
    neuron.previousDeltaWeights.pushBack(deltaWeights);
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
